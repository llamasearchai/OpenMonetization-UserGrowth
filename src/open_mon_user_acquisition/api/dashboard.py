"""Real-time dashboard API for OpenMonetization-UserAcquisition."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..orchestrator import WorkflowOrchestrator
from ..experiments import ABTestingManager
from ..observability import MetricsCollector


# Global instances
orchestrator: Optional[WorkflowOrchestrator] = None
ab_testing_manager: Optional[ABTestingManager] = None
metrics_collector: Optional[MetricsCollector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global orchestrator, ab_testing_manager, metrics_collector

    # Initialize components
    try:
        from ..config import ConfigManager
        from ..llm import LLMFallbackManager
        from ..storage import SQLiteStorageBackend

        config = ConfigManager()
        storage = SQLiteStorageBackend()
        llm_backend = LLMFallbackManager()

        orchestrator = WorkflowOrchestrator(
            config=config,
            storage=storage,
            llm_backend=llm_backend
        )
        await orchestrator.initialize()

        ab_testing_manager = ABTestingManager(storage=storage)
        await ab_testing_manager.initialize()

        metrics_collector = MetricsCollector()
        await metrics_collector.initialize()

        print("✅ Dashboard API initialized successfully")

    except Exception as e:
        print(f"❌ Failed to initialize dashboard API: {e}")
        # Continue with limited functionality

    yield

    # Shutdown components
    if orchestrator:
        await orchestrator.shutdown()
    if ab_testing_manager:
        await ab_testing_manager.shutdown()
    if metrics_collector:
        await metrics_collector.shutdown()

    print("✅ Dashboard API shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="OpenMonetization-UserAcquisition Dashboard",
    description="Real-time dashboard for user acquisition campaign monitoring and optimization",
    version="1.0.0",
    lifespan=lifespan
)


# Pydantic models for API responses
class SystemStatus(BaseModel):
    """System status response model."""
    initialized: bool
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    workflow_metrics: Optional[Dict[str, Any]] = None


class WorkflowSummary(BaseModel):
    """Workflow summary model."""
    id: str
    name: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    tasks_count: int


class ExperimentSummary(BaseModel):
    """Experiment summary model."""
    id: str
    name: str
    campaign_id: str
    status: str
    primary_metric: str
    total_sample_size: int
    target_sample_size: int
    variants_count: int
    winner: Optional[str] = None


class MetricData(BaseModel):
    """Metric data model."""
    name: str
    value: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML page."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OMUA Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .animate-pulse { animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <header class="mb-8">
                <h1 class="text-3xl font-bold text-gray-800 mb-2">
                    🚀 OpenMonetization-UserAcquisition Dashboard
                </h1>
                <p class="text-gray-600">Real-time monitoring and optimization platform</p>
            </header>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
                <!-- System Status -->
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">System Status</h2>
                    <div id="system-status" class="space-y-2">
                        <div class="animate-pulse text-gray-500">Loading...</div>
                    </div>
                </div>

                <!-- Active Workflows -->
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Active Workflows</h2>
                    <div id="active-workflows" class="space-y-2">
                        <div class="animate-pulse text-gray-500">Loading...</div>
                    </div>
                </div>

                <!-- A/B Experiments -->
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">A/B Experiments</h2>
                    <div id="ab-experiments" class="space-y-2">
                        <div class="animate-pulse text-gray-500">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- Metrics Charts -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">Workflow Performance</h2>
                    <canvas id="workflow-chart" width="400" height="200"></canvas>
                </div>

                <div class="bg-white rounded-lg shadow-md p-6">
                    <h2 class="text-xl font-semibold mb-4 text-gray-800">System Metrics</h2>
                    <canvas id="metrics-chart" width="400" height="200"></canvas>
                </div>
            </div>

            <!-- Recent Activity -->
            <div class="bg-white rounded-lg shadow-md p-6">
                <h2 class="text-xl font-semibold mb-4 text-gray-800">Recent Activity</h2>
                <div id="recent-activity" class="space-y-2 max-h-64 overflow-y-auto">
                    <div class="animate-pulse text-gray-500">Loading recent activity...</div>
                </div>
            </div>
        </div>

        <script>
            let workflowChart, metricsChart;

            async function updateDashboard() {
                try {
                    // Update system status
                    const statusResponse = await fetch('/api/status');
                    const statusData = await statusResponse.json();
                    updateSystemStatus(statusData);

                    // Update workflows
                    const workflowsResponse = await fetch('/api/workflows?limit=5');
                    const workflowsData = await workflowsResponse.json();
                    updateWorkflows(workflowsData.workflows || []);

                    // Update experiments
                    const experimentsResponse = await fetch('/api/experiments?limit=5');
                    const experimentsData = await experimentsResponse.json();
                    updateExperiments(experimentsData.experiments || []);

                    // Update metrics
                    const metricsResponse = await fetch('/api/metrics?limit=20');
                    const metricsData = await metricsResponse.json();
                    updateMetrics(metricsData.metrics || []);

                } catch (error) {
                    console.error('Dashboard update error:', error);
                }
            }

            function updateSystemStatus(data) {
                const statusDiv = document.getElementById('system-status');
                if (data.error) {
                    statusDiv.innerHTML = `<div class="text-red-500">❌ ${data.error}</div>`;
                    return;
                }

                let html = '';
                html += `<div class="flex items-center ${data.initialized ? 'text-green-500' : 'text-red-500'}">`;
                html += data.initialized ? '✅ System Online' : '❌ System Offline';
                html += '</div>';

                if (data.workflow_metrics) {
                    const metrics = data.workflow_metrics;
                    html += `<div class="text-sm text-gray-600 mt-2">`;
                    html += `Active: ${metrics.active_workflows || 0} | `;
                    html += `Pending: ${metrics.pending_workflows || 0} | `;
                    html += `Completed: ${metrics.completed_workflows || 0}`;
                    html += `</div>`;
                }

                statusDiv.innerHTML = html;
            }

            function updateWorkflows(workflows) {
                const workflowsDiv = document.getElementById('active-workflows');
                if (workflows.length === 0) {
                    workflowsDiv.innerHTML = '<div class="text-gray-500">No active workflows</div>';
                    return;
                }

                let html = '';
                workflows.forEach(workflow => {
                    const statusColor = getStatusColor(workflow.status);
                    html += `<div class="flex justify-between items-center p-2 bg-gray-50 rounded">`;
                    html += `<span class="font-medium">${workflow.name}</span>`;
                    html += `<span class="px-2 py-1 rounded text-xs ${statusColor}">${workflow.status}</span>`;
                    html += `</div>`;
                });
                workflowsDiv.innerHTML = html;
            }

            function updateExperiments(experiments) {
                const experimentsDiv = document.getElementById('ab-experiments');
                if (experiments.length === 0) {
                    experimentsDiv.innerHTML = '<div class="text-gray-500">No active experiments</div>';
                    return;
                }

                let html = '';
                experiments.forEach(experiment => {
                    const statusColor = getStatusColor(experiment.status);
                    html += `<div class="flex justify-between items-center p-2 bg-gray-50 rounded">`;
                    html += `<span class="font-medium">${experiment.name}</span>`;
                    html += `<span class="px-2 py-1 rounded text-xs ${statusColor}">${experiment.status}</span>`;
                    html += `</div>`;
                });
                experimentsDiv.innerHTML = html;
            }

            function updateMetrics(metrics) {
                const activityDiv = document.getElementById('recent-activity');
                if (metrics.length === 0) {
                    activityDiv.innerHTML = '<div class="text-gray-500">No recent metrics</div>';
                    return;
                }

                let html = '';
                metrics.slice(0, 10).forEach(metric => {
                    const time = new Date(metric.timestamp).toLocaleTimeString();
                    html += `<div class="text-sm text-gray-600 p-1">`;
                    html += `${time} - ${metric.name}: ${metric.value}`;
                    html += `</div>`;
                });
                activityDiv.innerHTML = html;

                // Update charts (simplified)
                updateCharts(metrics);
            }

            function updateCharts(metrics) {
                // Initialize charts if not exists
                if (!workflowChart) {
                    const workflowCtx = document.getElementById('workflow-chart').getContext('2d');
                    workflowChart = new Chart(workflowCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Workflow Executions',
                                data: [],
                                borderColor: 'rgb(59, 130, 246)',
                                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                }

                if (!metricsChart) {
                    const metricsCtx = document.getElementById('metrics-chart').getContext('2d');
                    metricsChart = new Chart(metricsCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Metric Values',
                                data: [],
                                backgroundColor: 'rgba(16, 185, 129, 0.6)',
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: { beginAtZero: true }
                            }
                        }
                    });
                }

                // Update chart data (simplified example)
                const labels = metrics.slice(-10).map(m => new Date(m.timestamp).toLocaleTimeString());
                const values = metrics.slice(-10).map(m => m.value);

                workflowChart.data.labels = labels;
                workflowChart.data.datasets[0].data = values;
                workflowChart.update();

                metricsChart.data.labels = labels;
                metricsChart.data.datasets[0].data = values;
                metricsChart.update();
            }

            function getStatusColor(status) {
                switch (status.toLowerCase()) {
                    case 'running': return 'bg-blue-100 text-blue-800';
                    case 'completed': return 'bg-green-100 text-green-800';
                    case 'failed': return 'bg-red-100 text-red-800';
                    case 'pending': return 'bg-yellow-100 text-yellow-800';
                    default: return 'bg-gray-100 text-gray-800';
                }
            }

            // Update dashboard every 5 seconds
            updateDashboard();
            setInterval(updateDashboard, 5000);

            // Initial load
            document.addEventListener('DOMContentLoaded', updateDashboard);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get system status information."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        status = await orchestrator.get_system_status()
        return SystemStatus(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@app.get("/api/workflows", response_model=Dict[str, List[WorkflowSummary]])
async def get_workflows(limit: int = Query(50, ge=1, le=100)):
    """Get workflow information."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        workflows = await orchestrator.list_active_workflows()

        # Convert to summary format
        workflow_summaries = []
        for workflow in workflows[:limit]:
            summary = WorkflowSummary(
                id=workflow.id,
                name=workflow.name,
                status=workflow.status,
                created_at=workflow.created_at,
                started_at=workflow.started_at,
                completed_at=workflow.completed_at,
                tasks_count=len(workflow.tasks)
            )
            workflow_summaries.append(summary)

        return {"workflows": workflow_summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflows: {str(e)}")


@app.get("/api/experiments", response_model=Dict[str, List[ExperimentSummary]])
async def get_experiments(
    campaign_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100)
):
    """Get A/B testing experiments."""
    if not ab_testing_manager:
        raise HTTPException(status_code=503, detail="A/B testing not available")

    try:
        experiments = await ab_testing_manager.list_experiments(
            campaign_id=campaign_id
        )

        # Convert to summary format
        experiment_summaries = []
        for experiment in experiments[:limit]:
            summary = ExperimentSummary(
                id=experiment.id,
                name=experiment.name,
                campaign_id=experiment.campaign_id,
                status=experiment.status.value,
                primary_metric=experiment.primary_metric,
                total_sample_size=sum(v.sample_size for v in experiment.variants),
                target_sample_size=experiment.target_sample_size,
                variants_count=len(experiment.variants),
                winner=experiment.winner_variant_id
            )
            experiment_summaries.append(summary)

        return {"experiments": experiment_summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiments: {str(e)}")


@app.get("/api/metrics", response_model=Dict[str, List[MetricData]])
async def get_metrics(
    metric_name: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """Get system metrics."""
    if not metrics_collector:
        # Return mock metrics if collector not available
        mock_metrics = [
            MetricData(
                name="workflow_created",
                value=1.0,
                timestamp=datetime.now(),
                metadata={"source": "mock"}
            ),
            MetricData(
                name="task_completed",
                value=1.0,
                timestamp=datetime.now(),
                metadata={"source": "mock"}
            )
        ]
        return {"metrics": mock_metrics[:limit]}

    try:
        # In a real implementation, this would query the metrics collector
        # For now, return empty list
        return {"metrics": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/api/workflow/{workflow_id}")
async def get_workflow_details(workflow_id: str):
    """Get detailed information about a specific workflow."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        workflow = await orchestrator.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return workflow.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow details: {str(e)}")


@app.get("/api/experiment/{experiment_id}")
async def get_experiment_details(experiment_id: str):
    """Get detailed information about a specific experiment."""
    if not ab_testing_manager:
        raise HTTPException(status_code=503, detail="A/B testing not available")

    try:
        results = await ab_testing_manager.get_experiment_results(experiment_id)
        if not results:
            raise HTTPException(status_code=404, detail="Experiment not found")

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment details: {str(e)}")


@app.post("/api/workflow/{workflow_id}/execute")
async def execute_workflow(workflow_id: str):
    """Execute a workflow."""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        workflow = await orchestrator.execute_workflow(workflow_id)
        return {"status": "executed", "workflow": workflow.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")


@app.post("/api/experiment/{experiment_id}/start")
async def start_experiment(experiment_id: str):
    """Start an A/B testing experiment."""
    if not ab_testing_manager:
        raise HTTPException(status_code=503, detail="A/B testing not available")

    try:
        experiment = await ab_testing_manager.start_experiment(experiment_id)
        return {"status": "started", "experiment": experiment.get_experiment_summary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start experiment: {str(e)}")


@app.post("/api/experiment/{experiment_id}/stop")
async def stop_experiment(experiment_id: str):
    """Stop an A/B testing experiment."""
    if not ab_testing_manager:
        raise HTTPException(status_code=503, detail="A/B testing not available")

    try:
        experiment = await ab_testing_manager.stop_experiment(experiment_id)
        return {"status": "stopped", "experiment": experiment.get_experiment_summary()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop experiment: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "orchestrator": orchestrator is not None,
            "ab_testing": ab_testing_manager is not None,
            "metrics": metrics_collector is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "dashboard:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
