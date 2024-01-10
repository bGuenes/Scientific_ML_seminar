from manim import *
from manim_slides import Slide
import autograd.numpy as np
import autograd

q = np.load("../data/pendulum/x.npy")
p = np.load("../data/pendulum/y.npy")
t = np.load("../data/pendulum/t.npy")

class S_test(Slide):
    def construct(self):
        self.camera.background_color = "#ECE7E2"

        def hamiltonian(x):
            q, p = np.split(x,2)
            return 3*(1-np.cos(q)) + p**2

        func = lambda x: autograd.grad(hamiltonian)(np.array([x[0], x[1]])) @ np.array([[0,-1], [1,0]])

        axis = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], axis_config={"color": BLACK}, x_length=6, y_length=6)
        VF = ArrowVectorField(func, x_range=[-2.5, 2.5, 0.4], y_range=[-2.5, 2.5, 0.4], colors=["#ABF5D1", "#87C2A5", "#456354"])

        x0, y0 = 0.6*q[0]+4, 0.6*p[0]
        p_dot = Dot([x0, y0, 0], color=RED, radius=0.05)
        p_dot_trace = TracedPath(p_dot.get_center, stroke_color=RED)

        field = VGroup(axis, VF).shift(4*RIGHT).scale(0.6)

        self.play(Create(axis))
        self.play(*[GrowArrow(i) for i in VF])
        self.wait(1)
"""
        self.play(Create(p_dot))
        self.add(p_dot_trace)
        for i in range(1, 200):
            x, y = 0.6*q[i]+4, 0.6*p[i]
            self.play(p_dot.animate.move_to([x, y, 0]), run_time=0.01)

        self.wait(2)
"""
