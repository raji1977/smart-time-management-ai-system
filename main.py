"""
Smart Time Management System
Single-file Flask + Mesa prototype with enhanced UI

Requirements:
- Python 3.8+
- pip install mesa flask pandas

Run:
    pip install -r requirements.txt
    python smart_time_management_app.py

Open http://127.0.0.1:5000

This file implements the Mesa multi-agent model and a Flask UI with Bootstrap + Chart.js charts
showing productivity metrics (tasks remaining over time, focus distribution, reminders, and task list).
"""

from flask import Flask, render_template_string, jsonify, request
from mesa import Agent, Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
import threading
import time
import uuid
import pandas as pd
from collections import deque, defaultdict

# ------------------------
# Core simulation classes
# ------------------------

class Task:
    def __init__(self, title, est_minutes, priority=3, deadline=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.est_minutes = est_minutes
        self.priority = priority  # 1 (highest) - 5 (lowest)
        self.deadline = deadline
        self.assigned_slot = None
        self.time_spent = 0
        self.completed = False

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'est_minutes': self.est_minutes,
            'priority': self.priority,
            'deadline': self.deadline,
            'assigned_slot': self.assigned_slot,
            'time_spent': self.time_spent,
            'completed': self.completed
        }

class UserModel(Model):
    """Mesa model coordinating agents and tasks."""
    def __init__(self, tasks=None):
        super().__init__()
        self.schedule = BaseScheduler(self)
        self.time_minute = 0  # simulated minutes passed in the day
        self.tasks = tasks or []
        self.reminders = deque()
        self.focus_log = []

        # Create agents
        self.scheduler_agent = SchedulerAgent('scheduler', self)
        self.optimizer_agent = TaskOptimizerAgent('optimizer', self)
        self.reminder_agent = ReminderAgent('reminder', self)
        self.focus_agent = FocusMonitorAgent('focus', self)
        self.report_agent = ReportGeneratorAgent('report', self)

        for a in [self.scheduler_agent, self.optimizer_agent, self.reminder_agent, self.focus_agent, self.report_agent]:
            self.schedule.add(a)

        # Data collector (simple)
        self.datacollector = DataCollector(
            model_reporters={
                'time_minute': lambda m: m.time_minute,
                'tasks_remaining': lambda m: len([t for t in m.tasks if not t.completed]),
                'tasks_completed': lambda m: len([t for t in m.tasks if t.completed])
            }
        )

    def step(self):
        # One model step will represent 5 simulated minutes
        self.time_minute += 5
        self.schedule.step()
        self.datacollector.collect(self)

    def to_json(self):
        return {
            'time_minute': self.time_minute,
            'tasks': [t.to_dict() for t in self.tasks],
            'reminders': list(self.reminders),
            'focus_log': list(self.focus_log)
        }

# ------------------------
# Agent implementations
# ------------------------

class SchedulerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # Basic scheduling: sort by priority then earliest deadline
        unscheduled = [t for t in self.model.tasks if t.assigned_slot is None and not t.completed]
        unscheduled.sort(key=lambda x: (x.priority, x.deadline or '9999'))
        # Assign in contiguous slots (each slot = 30 minutes) until day end (480 minutes)
        # Try to respect existing assigned tasks
        used = []
        for t in self.model.tasks:
            if t.assigned_slot is not None and not t.completed:
                used.append((t.assigned_slot, t.assigned_slot + max(30, t.est_minutes)))
        current_minute = max(0, self.model.time_minute)
        for t in unscheduled:
            # find first gap starting from current_minute
            slot = current_minute
            # naive: just assign at current_minute
            t.assigned_slot = slot
            current_minute += max(30, t.est_minutes)

class TaskOptimizerAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # Simple optimizer: if many tasks remaining, bump priority of short tasks
        remaining = [t for t in self.model.tasks if not t.completed]
        if not remaining:
            return
        avg_est = sum(t.est_minutes for t in remaining) / len(remaining)
        for t in remaining:
            if t.est_minutes < avg_est * 0.6 and len(remaining) > 3:
                t.priority = max(1, t.priority - 1)

class ReminderAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # Remind tasks whose assigned slot is near (within 30 simulated minutes)
        now = self.model.time_minute
        for t in self.model.tasks:
            if not t.completed and t.assigned_slot is not None:
                if 0 <= t.assigned_slot - now <= 30:
                    text = f"Reminder: '{t.title}' starts at minute {t.assigned_slot}."
                    if text not in self.model.reminders:
                        self.model.reminders.appendleft(text)
                        while len(self.model.reminders) > 50:
                            self.model.reminders.pop()

class FocusMonitorAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.current_task_id = None
        self.task_timer = 0

    def step(self):
        # Work on the task scheduled now
        now = self.model.time_minute
        for t in self.model.tasks:
            if t.assigned_slot is not None and not t.completed:
                start = t.assigned_slot
                end = t.assigned_slot + t.est_minutes
                if start <= now < end:
                    t.time_spent += 5
                    self.model.focus_log.append({'minute': now, 'task': t.title, 'worked': 5})
                    if t.time_spent >= t.est_minutes:
                        t.completed = True
                        self.model.reminders.appendleft(f"Completed: {t.title}")
                    return
        # idle
        self.model.focus_log.append({'minute': now, 'task': None, 'worked': 0})

class ReportGeneratorAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # placeholder - heavy reports generated on demand
        return

# ------------------------
# Flask app + threading
# ------------------------

app = Flask(__name__)
MODEL = None
SIM_THREAD = None
SIM_LOCK = threading.Lock()
RUNNING = False

# Enhanced HTML with Bootstrap and Chart.js
HTML = '''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Smart Time Management — Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body{padding:20px; background: linear-gradient(120deg,#f6f8ff,#fff)}
    .card{border-radius:12px; box-shadow:0 6px 18px rgba(50,50,93,0.08)}
    .task-badge{font-size:0.8rem}
    .small-muted{font-size:0.85rem; color:#666}
  </style>
</head>
<body>
  <div class="container">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <h2>Smart Time Management</h2>
      <div>
        <button class="btn btn-success me-1" onclick="startSim()">Start</button>
        <button class="btn btn-danger me-1" onclick="stopSim()">Stop</button>
        <button class="btn btn-primary me-1" onclick="stepSim()">Step</button>
        <button class="btn btn-outline-secondary" onclick="resetSim()">Reset</button>
      </div>
    </div>

    <div class="row g-3">
      <div class="col-md-4">
        <div class="card p-3">
          <h5>Overview</h5>
          <p class="small-muted">Model time: <span id="time">-</span> minutes</p>
          <p class="small-muted">Tasks remaining: <span id="tasks_remaining">-</span></p>
          <p class="small-muted">Tasks completed: <span id="tasks_completed">-</span></p>
          <div id="task-list" style="max-height:240px; overflow:auto; margin-top:8px"></div>
        </div>

        <div class="card p-3 mt-3">
          <h6>Reminders</h6>
          <ul id="reminders" class="list-unstyled small-muted" style="max-height:180px; overflow:auto"></ul>
        </div>
      </div>

      <div class="col-md-8">
        <div class="card p-3 mb-3">
          <h5>Productivity Over Time</h5>
          <canvas id="lineChart" height="120"></canvas>
        </div>

        <div class="card p-3 mb-3">
          <div class="row">
            <div class="col-md-6">
              <h5>Focus Distribution</h5>
              <canvas id="donutChart" height="180"></canvas>
            </div>
            <div class="col-md-6">
              <h5>Focus Log (recent)</h5>
              <ul id="focus_log" class="list-group" style="max-height:260px; overflow:auto"></ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <footer class="mt-4 small text-muted">Prototype — demo only</footer>
  </div>

<script>
let lineChart=null, donutChart=null;
async function fetchState(){
  const r = await fetch('/state');
  const j = await r.json();
  document.getElementById('time').innerText = j.time_minute;
  document.getElementById('tasks_remaining').innerText = j.tasks.filter(t=>!t.completed).length;
  document.getElementById('tasks_completed').innerText = j.tasks.filter(t=>t.completed).length;

  // task list
  const tl = document.getElementById('task-list'); tl.innerHTML='';
  j.tasks.forEach(t=>{
    const div = document.createElement('div'); div.className='mb-2';
    div.innerHTML = `<div class='d-flex justify-content-between'><div><strong>${t.title}</strong><div class='small-muted'>${t.est_minutes}m • priority ${t.priority}</div></div><div class='text-end'>${t.assigned_slot===null?'<span class="badge bg-secondary">unscheduled</span>':'<span class="badge bg-info task-badge">slot '+t.assigned_slot+'</span>'}${t.completed?'<div class="text-success">✓ done</div>':''}</div></div>`;
    tl.appendChild(div);
  });

  // reminders
  const rem = document.getElementById('reminders'); rem.innerHTML='';
  j.reminders.slice(0,20).forEach(rm=>{ const li=document.createElement('li'); li.innerText=rm; rem.appendChild(li); });

  // focus log
  const flog = document.getElementById('focus_log'); flog.innerHTML='';
  j.focus_log.slice(-50).reverse().forEach(fl=>{ const li=document.createElement('li'); li.className='list-group-item list-group-item-light'; li.innerText = `${fl.minute}m — ${fl.task || 'idle'} (+${fl.worked}m)`; flog.appendChild(li); });
}

async function fetchCharts(){
  const r = await fetch('/chart-data');
  const d = await r.json();

  // line chart: tasks_remaining
  const ctx = document.getElementById('lineChart').getContext('2d');
  if(lineChart) lineChart.destroy();
  lineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [{
        label: 'Tasks Remaining',
        data: d.tasks_remaining,
        tension: 0.3,
        fill: true
      },{
        label: 'Tasks Completed',
        data: d.tasks_completed,
        tension: 0.3
      }]
    },
    options: {responsive:true, plugins:{legend:{position:'top'}}}
  });

  // donut: focus distribution
  const ctx2 = document.getElementById('donutChart').getContext('2d');
  if(donutChart) donutChart.destroy();
  donutChart = new Chart(ctx2, {
    type: 'doughnut',
    data: {
      labels: d.focus_labels,
      datasets: [{data: d.focus_values}]
    },
    options: {responsive:true, plugins:{legend:{position:'right'}}}
  });
}

let poll=null;
function startPolling(){ if(!poll) poll = setInterval(()=>{ fetchState(); fetchCharts(); }, 1200); }
startPolling();

async function startSim(){ await fetch('/start', {method:'POST'}); }
async function stopSim(){ await fetch('/stop', {method:'POST'}); }
async function stepSim(){ await fetch('/step', {method:'POST'}); await fetchState(); await fetchCharts(); }
async function resetSim(){ await fetch('/reset', {method:'POST'}); await fetchState(); await fetchCharts(); }

// initial draw
fetchState(); fetchCharts();
</script>
</body>
</html>
'''

# Helper to create demo tasks
def sample_tasks():
    return [
        Task('Email triage', 20, priority=2),
        Task('Write report section', 90, priority=3),
        Task('Code review', 45, priority=2),
        Task('Deep work: model training', 120, priority=1),
        Task('Plan next day', 30, priority=4)
    ]

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/start', methods=['POST'])
def start():
    global MODEL, SIM_THREAD, RUNNING
    with SIM_LOCK:
        if MODEL is None:
            MODEL = UserModel(tasks=sample_tasks())
        if RUNNING:
            return ('', 204)
        RUNNING = True
        SIM_THREAD = threading.Thread(target=run_sim_loop, daemon=True)
        SIM_THREAD.start()
    return ('', 204)

@app.route('/stop', methods=['POST'])
def stop():
    global RUNNING
    with SIM_LOCK:
        RUNNING = False
    return ('', 204)

@app.route('/step', methods=['POST'])
def step_once():
    global MODEL
    with SIM_LOCK:
        if MODEL is None:
            MODEL = UserModel(tasks=sample_tasks())
        MODEL.step()
    return ('', 204)

@app.route('/reset', methods=['POST'])
def reset():
    global MODEL, RUNNING
    with SIM_LOCK:
        MODEL = UserModel(tasks=sample_tasks())
        RUNNING = False
    return ('', 204)

@app.route('/state')
def state():
    global MODEL
    if MODEL is None:
        return jsonify({'error':'no model initialized'}), 400
    return jsonify(MODEL.to_json())

@app.route('/report')
def report():
    global MODEL
    if MODEL is None:
        return jsonify({'error':'no model initialized'}), 400
    df = pd.DataFrame([t.to_dict() for t in MODEL.tasks])
    if df.empty:
        return jsonify({'error':'no tasks'}), 400
    completed = df[df['completed']]
    pct_done = 0 if len(df)==0 else round(len(completed)/len(df)*100,2)
    avg_effort = round(df['est_minutes'].mean(),2)
    report = {
        'total_tasks': len(df),
        'completed': len(completed),
        'pct_done': pct_done,
        'avg_est_minutes': avg_effort,
    }
    return jsonify(report)

@app.route('/chart-data')
def chart_data():
    """Return data for charts: tasks remaining/completed over time and focus distribution."""
    global MODEL
    if MODEL is None:
        return jsonify({'error':'no model initialized'}), 400

    # Get time series from datacollector if available
    try:
        df = MODEL.datacollector.get_model_vars_dataframe()
        labels = [str(int(x)) + 'm' for x in df['time_minute'].tolist()]
        tasks_remaining = df['tasks_remaining'].tolist()
        tasks_completed = df['tasks_completed'].tolist()
    except Exception:
        labels = []
        tasks_remaining = []
        tasks_completed = []

    # Focus distribution: sum worked minutes per task
    focus = defaultdict(int)
    for e in MODEL.focus_log:
        if e['task']:
            focus[e['task']] += e['worked']
        else:
            focus['Idle'] += e['worked']
    focus_items = sorted(focus.items(), key=lambda x: -x[1])
    focus_labels = [k for k,v in focus_items]
    focus_values = [v for k,v in focus_items]

    return jsonify({
        'labels': labels,
        'tasks_remaining': tasks_remaining,
        'tasks_completed': tasks_completed,
        'focus_labels': focus_labels,
        'focus_values': focus_values
    })

# Simulation loop runs in background thread when started

def run_sim_loop():
    global MODEL, RUNNING
    while True:
        with SIM_LOCK:
            if not RUNNING or MODEL is None:
                pass
            else:
                MODEL.step()
        time.sleep(0.9)

if __name__ == '__main__':
    print('Starting Flask app — open http://127.0.0.1:5000')
    app.run(debug=True)
