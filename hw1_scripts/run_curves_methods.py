from src import PROJECT_ROOT

import os
import glob
import src.geometry.curves as curves

data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
file = glob.glob(os.path.join(data_dir, '*.off'))[0]


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
interval = curves.Astroid().get_interval(start=start, end=end, n_points=n_points)
save_path = os.path.join(PROJECT_ROOT, 'data', 'images')
os.makedirs(save_path, exist_ok=True)
save_path = os.path.join(save_path, 'Astroid.png')

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

# Sweep over all parameters with a specific curve
curves.sweep_curve(
    curve_obj=curves.Astroid, interval=interval, params=params,
    title="Astroid", save_path=save_path)
