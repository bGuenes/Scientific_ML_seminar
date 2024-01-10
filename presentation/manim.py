from manim import *
from manim_slides import Slide
import autograd.numpy as np
import autograd

# Load data
q = np.load("../data/pendulum/x.npy")
p = np.load("../data/pendulum/y.npy")

class S1(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = "#ECE7E2"

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # Create title
        title = Text("Pendulum", font="Sans", font_size=60, color=BLACK)

        # Create pendulum components
        l = -3
        start = [0, 0, 0]
        end = [0, l, 0]

        pivot = Dot(start, color=BLACK)
        rod = Line(start, end, color=BLACK)
        bob = Dot(end, color="#94424F", radius=0.2)

        pendulum = VGroup(pivot, rod, bob)

        # Crete phase space
        def hamiltonian(x):
            q, p = np.split(x,2)
            return 3*(1-np.cos(q)) + p**2

        func = lambda x: autograd.grad(hamiltonian)(np.array([x[0], x[1]])) @ np.array([[0,-1], [1,0]])

        axis = Axes(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1], axis_config={"color": BLACK}, x_length=7, y_length=7)
        VF = ArrowVectorField(func, x_range=[-2.5, 2.5, 0.4], y_range=[-2.5, 2.5, 0.4], colors=["#ABF5D1", "#87C2A5", "#456354"])

        field = VGroup(axis, VF).shift(4*RIGHT).scale(0.7)

        x0, y0 = 0.6 * q[0] + 4, 0.6 * p[0]
        p_dot = Dot([x0, y0, 0], color="#94424F", radius=0.05)
        p_dot_trace = TracedPath(p_dot.get_center, stroke_color="#94424F")

        # -----------------------------------
        # Animations slide 1
        self.play(Write(title))
        self.next_slide()

        self.play(title.animate.to_corner(UL))
        self.play(Create(rod), Create(pivot), Create(bob))
        self.wait(1)

        self.play(pendulum.animate.shift(3.5*LEFT))
        self.play(Create(axis))
        self.play(*[GrowArrow(i) for i in VF])
        self.wait(1)

        self.play(Rotate(pendulum, q[0], about_point=[-3.5, 0, 0]), run_time=1)
        self.play(Create(p_dot))
        self.add(p_dot_trace)

        # Animate pendulum using existing data
        self.next_slide(loop=True)
        for i in range(1, 200):
            angle = q[i] - q[i-1]
            self.play(Rotate(pendulum, angle, about_point=[-3.5, 0, 0]), run_time=0.015)

            x, y = 0.6 * q[i] + 4, 0.6 * p[i]
            self.play(p_dot.animate.move_to([x, y, 0]), run_time=0.01)
        self.next_slide()

        # /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # -----------------------------------


