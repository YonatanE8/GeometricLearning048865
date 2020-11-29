from src import PROJECT_ROOT
from src.visualizations.plot_curves import (plot_geometric_flow,
                                            plot_curves_normals_vs_tangent)

import os
import src.geometry.curves as curves


# --- Plot Curves
data_dir = os.path.join(PROJECT_ROOT, 'data', 'images')
os.makedirs(data_dir, exist_ok=True)

a = 1.
b = 1.
c = 1.
curves_list = [
curves.Astroid(a=a),
curves.Cardioid(a=a),
curves.Conchoid(a=a),
curves.Epicycloid(a=a, b=b),
curves.Epitrochoid(a=a, b=b, c=c),
curves.DescartesFolium(a=a),
curves.Hypocycloid(a=a, b=b),
curves.Hypotrochoid(a=a, b=b, c=c),
curves.InvoluteCircle(a=a)
]

# --- Sweep over all parameters with a specific curve
titles = [
f"Astroid: a = {a}",
f"Cardioid: a = {a}",
f"Conchoid: a = {a}",
f"Epicycloid: a = {a}, b = {b}",
f"Epitrochoid: a = {a}, b = {b}, c = {c}",
f"DescartesFolium: a = {a}",
f"Hypocycloid: a = {a}, b = {b}",
f"Hypotrochoid: a = {a}, b = {b}, c = {c}",
f"InvoluteCircle: a = {a}",
]

start = -20.
end = 20.
n_points = 20000
interval = curves.Knot().get_interval(start=start, end=end, n_points=n_points)
save_path = os.path.join(data_dir, 'Cusp.png')

a_params = [-5, -2, -1, -0.5, 0.5, 1, 2, 5]
b_params = [-5, -2, -1, -0.5, 0.5, 1, 2, 5]
c_params = [-5, -2, -1, -0.5, 0.5, 1, 2, 5]

# Set parameters to sweep over, depending on the curve
params = [
    {'a': a} for a in a_params
]
# params = [
#     {'a': a, 'b': b} for a in a_params for b in b_params
# ]
# params = [
#     {'a': a, 'b': b, 'c': c} for a in a_params for b in b_params for c in c_params
# ]

# curves.sweep_curve(
#     curve_obj=curves.Cusp, interval=interval, params=params,
#     title="Cusp", save_path=save_path)

start = 0
end = 100
n_points = 100
a = 2
b = 4
c = 2
# curve_obj = curves.Cusp()
# curve_obj = curves.Cardioid(a=a)
# curve_obj = curves.Astroid(a=a)
# curve_obj = curves.DescartesFolium(a=a)
# curve_obj = curves.InvoluteCircle(a=a)
# curve_obj = curves.Epicycloid(a=a, b=b)
curve_obj = curves.Ellipse(a=a, b=b)
# curve_obj = curves.Hypotrochoid(a=a, b=b, c=c)
interval = curve_obj.get_interval(start=start, end=end, n_points=n_points)

save_path = os.path.join(data_dir, 'EllipseGeometricFlow.png')
title = "Ellipse: Evolution Curve, Mean Curvature Flow & Arc Length vs. Time"
# plot_geometric_flow(curve_obj=curve_obj, interval=interval, title=title,
#                     save_path=save_path)

plot_curves_normals_vs_tangent(curve_obj=curve_obj, interval=interval)



















