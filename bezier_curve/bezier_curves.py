import numpy as np
from scipy.special import comb
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State
from shapely.geometry import Polygon, box, Point

# Constants
field_size = 144

# Define polygonal obstacles
obstacles = [
    Polygon([
        (48, 48), (96, 48), (96, 96), (48, 96)  # Submersible zone (center 48x48 area)
    ]),
    # You can add more polygons here if needed
]

default_pts = {
    'start_x': 56.0, 'start_y': 12.0,
    'mid_x': 34.0, 'mid_y': 94.0,
    'end_x': 11.0, 'end_y': 130.0,
    'start_angle': 0,
    'end_angle': 90,
    'rect_width': 18,
    'rect_height': 18,
    'width_offset': 2,
    'height_offset': 2,
}

animation_data = {'running': True, 't': 0}

def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, num_points=100):
    n = len(control_points) - 1
    t_vals = np.linspace(0, 1, num_points)
    curve = np.zeros((num_points, 2))
    for i in range(n + 1):
        bern = bernstein_poly(i, n, t_vals)[:, np.newaxis]
        curve += bern * control_points[i]
    return curve

def create_rectangle(center, width, height, angle_deg):
    angle = np.radians(angle_deg)
    dx = width / 2
    dy = height / 2
    corners = np.array([
        [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy], [-dx, -dy]
    ])
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = corners @ rot_matrix.T
    translated = rotated + center
    return translated[:, 0], translated[:, 1]

def is_overlapping(center, width, height, angle, obstacles, field_size):
    cx, cy = center
    dx = width / 2
    dy = height / 2
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    angle_rad = np.deg2rad(angle)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated = corners @ rot_matrix.T + np.array([cx, cy])
    robot_poly = Polygon(rotated)
    field_box = box(0, 0, field_size, field_size)

    return any(robot_poly.intersects(obs) for obs in obstacles) or not field_box.contains(robot_poly)

def adjust_midpoint_to_avoid(start, end, rect_w, rect_h, start_angle, end_angle, obstacles, field_size, preference='fastest', top_n=5):
    start = np.array(start)
    end = np.array(end)
    mid = (start + end) / 2
    offset_range = np.linspace(-80, 80, 50)

    best_shortest = (float('inf'), None)
    best_fastest = (float('inf'), None)
    valid_candidates = []

    for dx in offset_range:
        for dy in offset_range:
            test_mid = mid + np.array([dx, dy])
            ctrl_pts = [start, test_mid, end]
            path = bezier_curve(ctrl_pts, num_points=50)

            collisions = any(
                is_overlapping(pt, rect_w, rect_h,
                               start_angle + (end_angle - start_angle) * i / len(path),
                               obstacles, field_size)
                for i, pt in enumerate(path)
            )
            if not collisions:
                diffs = np.diff(path, axis=0)
                total_length = np.sum(np.linalg.norm(diffs, axis=1))
                angles = np.arctan2(diffs[:, 1], diffs[:, 0])
                angle_diff = np.diff(angles)
                angle_diff = np.mod(angle_diff + np.pi, 2 * np.pi) - np.pi
                curvature_penalty = np.sum(np.abs(angle_diff))
                fast_cost = total_length + 100 * curvature_penalty

                valid_candidates.append((total_length, fast_cost, test_mid.copy()))

                if total_length < best_shortest[0]:
                    best_shortest = (total_length, test_mid.copy())
                if fast_cost < best_fastest[0]:
                    best_fastest = (fast_cost, test_mid.copy())

    debug = f"🔍 Checked {len(offset_range)**2} candidates.\n"
    if valid_candidates:
        debug += f"✅ Found {len(valid_candidates)} valid midpoints.\n"
        valid_candidates.sort(key=lambda x: x[1])
        for i, (length, fast_cost, midpt) in enumerate(valid_candidates[:top_n]):
            debug += f"#{i+1}: midpoint={midpt.tolist()}, length={length:.2f}, fast_cost={fast_cost:.2f}\n"
    else:
        debug += "❌ No valid path found.\n"

    if preference == 'fastest' and best_fastest[1] is not None:
        debug += f"🚀 Selected FASTEST path.\n"
        return best_fastest[1].tolist(), debug
    elif preference == 'shortest' and best_shortest[1] is not None:
        debug += f"📏 Selected SHORTEST path.\n"
        return best_shortest[1].tolist(), debug
    elif best_fastest[1] is not None:
        debug += f"🚀 Fallback to FASTEST path.\n"
        return best_fastest[1].tolist(), debug
    elif best_shortest[1] is not None:
        debug += f"📏 Fallback to SHORTEST path.\n"
        return best_shortest[1].tolist(), debug
    else:
        debug += "⚠️ No valid path found. Using default midpoint.\n"
        return mid.tolist(), debug

# Dash app
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='bezier-plot', config={'displayModeBar': False}, style={'flex': '3'}),
        html.Div([
            html.H4("Debug Output"),
            html.Div(id='debug-output', style={
                'whiteSpace': 'pre-wrap',
                'color': 'green',
                'border': '1px solid #ccc',
                'padding': '10px',
                'height': '700px',
                'overflowY': 'auto',
                'fontFamily': 'monospace',
                'fontSize': '13px',
            })
        ], style={'flex': '1', 'padding': '10px'})
    ], style={'display': 'flex', 'flexDirection': 'row'}),

    html.Div([
        html.Label("Start (x, y)"),
        dcc.Input(id='start_x', type='number', value=default_pts['start_x'], style={'width': '80px'}),
        dcc.Input(id='start_y', type='number', value=default_pts['start_y'], style={'width': '80px'}),

        html.Label("Mid (x, y)"),
        dcc.Input(id='mid_x', type='number', value=default_pts['mid_x'], style={'width': '80px'}),
        dcc.Input(id='mid_y', type='number', value=default_pts['mid_y'], style={'width': '80px'}),

        html.Label("End (x, y)"),
        dcc.Input(id='end_x', type='number', value=default_pts['end_x'], style={'width': '80px'}),
        dcc.Input(id='end_y', type='number', value=default_pts['end_y'], style={'width': '80px'}),

        html.Label("Start Angle"),
        dcc.Input(id='start_angle', type='number', value=default_pts['start_angle'], style={'width': '80px'}),
        html.Label("End Angle"),
        dcc.Input(id='end_angle', type='number', value=default_pts['end_angle'], style={'width': '80px'}),

        html.Label("Width"),
        dcc.Input(id='rect_width', type='number', value=default_pts['rect_width'], style={'width': '80px'}),
        html.Label("Height"),
        dcc.Input(id='rect_height', type='number', value=default_pts['rect_height'], style={'width': '80px'}),

        html.Label("Path Preference"),
        dcc.Dropdown(
            id='path_preference',
            options=[{'label': 'Fastest', 'value': 'fastest'}, {'label': 'Shortest', 'value': 'shortest'}],
            value='fastest',
            clearable=False,
            style={'width': '150px'}
        ),

        html.Button("Auto-adjust Midpoint", id='adjust-mid-btn'),
        html.Button("Stop/Start Animation", id='toggle-animation'),
        dcc.Interval(id='interval', interval=100, n_intervals=0),
        dcc.Input(id='width_offset', type='number', value=0, step=1, placeholder='Width Offset'),
        dcc.Input(id='height_offset', type='number', value=0, step=1, placeholder='Height Offset'),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px', 'padding': '20px'})
])

@app.callback(
    Output('mid_x', 'value'),
    Output('mid_y', 'value'),
    Output('debug-output', 'children'),
    Input('adjust-mid-btn', 'n_clicks'),
    State('start_x', 'value'), State('start_y', 'value'),
    State('end_x', 'value'), State('end_y', 'value'),
    State('rect_width', 'value'), State('rect_height', 'value'),
    State('start_angle', 'value'), State('end_angle', 'value'),
    State('width_offset', 'value'), State('height_offset', 'value'),
    State('path_preference', 'value'),
    prevent_initial_call=True
)
def auto_adjust(n_clicks, sx, sy, ex, ey, w, h, sa, ea, wo, ho, preference):
    debug = "🔍 Starting path calculation...\n"
    best_mid, extra_debug = adjust_midpoint_to_avoid(
        (sx, sy), (ex, ey), w + wo, h + ho, sa, ea, obstacles, field_size, preference
    )
    debug += extra_debug

    if best_mid != [(sx + ex) / 2, (sy + ey) / 2]:
        debug += f"✅ Best midpoint selected: {best_mid}\n"
    else:
        debug += "⚠️ No better path found. Using default midpoint.\n"

    return best_mid[0], best_mid[1], debug

@app.callback(
    Output('bezier-plot', 'figure'),
    Input('interval', 'n_intervals'),
    Input('start_x', 'value'), Input('start_y', 'value'),
    Input('mid_x', 'value'), Input('mid_y', 'value'),
    Input('end_x', 'value'), Input('end_y', 'value'),
    Input('start_angle', 'value'), Input('end_angle', 'value'),
    Input('rect_width', 'value'), Input('rect_height', 'value'),
)
def update_curve(n_intervals, sx, sy, mx, my, ex, ey, sa, ea, w, h):
    control_pts = [np.array([sx, sy]), np.array([mx, my]), np.array([ex, ey])]
    curve = bezier_curve(control_pts)
    t = animation_data['t']
    if animation_data['running']:
        t = (t + 1) % len(curve)
        animation_data['t'] = t
    center = curve[t]
    angle = sa + (ea - sa) * t / (len(curve) - 1)
    rx, ry = create_rectangle(center, w, h, angle)
    collision = is_overlapping(center, w, h, angle, obstacles, field_size)
    color = 'red' if collision else 'blue'

    fig = go.Figure()

    # draw field image
    fig.add_layout_image(
        dict(
            source="assets/intothedeep.png",  # Ensure image is 144x144 in scale
            x=0,
            y=field_size,
            sizex=field_size,
            sizey=field_size,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch"
        )
    )

    # Draw obstacles
    for poly in obstacles:
        x, y = poly.exterior.xy
        fig.add_trace(
            go.Scatter(x=list(x), y=list(y), fill='toself', fillcolor='rgba(200,200,200,0.4)', line=dict(color='gray'),
                       name='Obstacle'))

    fig.add_trace(go.Scatter(x=curve[:, 0], y=curve[:, 1], mode='lines', name='Bézier'))
    fig.add_trace(go.Scatter(x=[pt[0] for pt in control_pts], y=[pt[1] for pt in control_pts],
                             mode='markers+lines', name='Control Points'))
    fig.add_trace(go.Scatter(x=rx, y=ry, mode='lines', line=dict(color=color, width=3), name='Robot'))

    fig.update_layout(
        xaxis=dict(range=[0, 144], scaleanchor='y', scaleratio=1, visible=False),
        yaxis=dict(range=[0, 144], visible=False),
        height=700, margin=dict(t=20, b=20, l=20, r=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

@app.callback(
    Output('interval', 'disabled'),
    Input('toggle-animation', 'n_clicks'),
)
def toggle_animation(n):
    animation_data['running'] = (n % 2 == 0)
    return not animation_data['running']

app.run(debug=True)
