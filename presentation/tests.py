from manim import *
from manim_slides import Slide, ThreeDSlide
import autograd.numpy as np
import autograd
from classes import *

q = np.load("../data/pendulum/x.npy")
p = np.load("../data/pendulum/y.npy")
t = np.load("../data/pendulum/t.npy")
q_HNN, p_HNN = np.load("../data/pendulum/xHNN.npy")

class test1(Slide):
    def construct(self):
        self.camera.background_color = "#ECE7E2"

        def hamiltonian(x):
            q, p = np.split(x,2)
            return 3*(1-autograd.numpy.cos(q)) + p**2

        func = lambda x: autograd.grad(hamiltonian)(np.array([x[0], x[1]])) @ np.array([[0,-1], [1,0]])

        axis = Axes(x_range=[-3, 3, 1], y_range=[-3, 3, 1], axis_config={"color": BLACK}, x_length=6, y_length=6)
        labels = axis.get_axis_labels(Tex("q", color=BLACK).scale(0.5), Tex("p", color=BLACK).scale(0.5))
        axis = VGroup(axis, labels)
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

class test2(Slide):
    def construct(self):
        self.camera.background_color = "#ECE7E2"

        title = Text("Hamiltonian Neural Networks", font="Helvetica", font_size=65, color=BLACK, weight=BOLD).move_to(UP)
        subtitle = Text("Seminar: Scientific Machine Learning", font="Helvetica", font_size=30, color=BLACK).next_to(title, DOWN)
        footer = Text("Berkay Günes", font="Helvetica", font_size=12, color=BLACK).to_edge(DL)
        date = Text("15.01.2024", font="Helvetica", font_size=12, color=BLACK).to_edge(DR)

        self.next_slide()
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.play(FadeIn(footer), FadeIn(date))
        self.next_slide()

class test3(Slide):
    def construct(self):
        self.camera.background_color = "#ECE7E2"

        title = Text("Learning with basic NN", font="Helvetica", font_size=60, color=BLACK).to_edge(UL)

        NN = NeuralNetworkMobject([2,5,2]).scale(1.3).move_to(2*DOWN)
        NN.label_inputs(["q", "p"])
        NN.label_outputs(["dq", "dp"])

        CArrow1 = CurvedArrow([2,-2,0], [2,1,0], angle=3, color=BLACK)
        text1 = Text("integrate", font="Helvetica", color=BLACK).move_to(UP)
        CArrow2 = CurvedArrow([-2, 1, 0], [-2, -2, 0], angle=3, color=BLACK)

        NN_circle = VGroup(NN, CArrow1, text1, CArrow2)

        self.play(Write(title), run_time=1)
        self.wait(1)
        self.play(Create(NN), run_time=3)
        self.wait(1)
        self.play(Create(CArrow1), run_time=1)
        self.play(Write(text1), run_time=1)
        self.play(Create(CArrow2), run_time=1)

        self.next_slide()
        self.wait(2)
        self.play(NN_circle.animate.scale(0.4).move_to([3*RIGHT+2.5*DOWN]), run_time=2)

class test4(ThreeDSlide):
    def construct(self):
        self.camera.background_color = "#ECE7E2"

        axes = ThreeDAxes(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1], z_range=[-4, 4, 1],
                          axis_config={"color": BLACK}, x_length=7, y_length = 7)
        labels = axes.get_axis_labels(Tex("q", color=BLACK).scale(0.5), Tex("p", color=BLACK).scale(0.5), Tex("H", color=BLACK).scale(0.5))
        axis = VGroup(axes, labels)

        HNN_model = MLP()
        HNN_model.load_state_dict(torch.load("../data/pendulum/model_HNN.pt"))
        HNN_model.eval()

        def func(x):
            x = torch.tensor(x[0:2], dtype=torch.float32, requires_grad=True)
            pred = HNN_model(x)
            pred = torch.autograd.grad(pred.sum(), x, create_graph=True)[0]
            dH = torch.zeros_like(pred)
            dH.T[0] = pred.T[1]
            dH.T[1] = -pred.T[0]
            return dH.detach().numpy()

        VF_HNN = ArrowVectorField(func, x_range=[-2.5, 2.5, 0.4], y_range=[-2.5, 2.5, 0.4],
                                  colors=["#ABF5D1", "#87C2A5", "#456354"])
        field_HNN = VGroup(axis, VF_HNN).shift(4 * RIGHT).scale(0.7)


        def out(x,y):
            coor = torch.tensor([x,y], dtype=torch.float32, requires_grad=True)
            pred = HNN_model(coor)
            z = 0.5*pred.detach().numpy().sum()
            return x, y, z

        x0, y0 = 0.7 * q_HNN[0] + 4, 0.7 * p_HNN[0]
        p_dot = Dot([x0, y0, 0], color="#94424F", radius=0.05)
        p_dot_trace = TracedPath(p_dot.get_center, stroke_color="#94424F")

        # Surface of Hamiltonian
        ham_surf = Surface(lambda x,y: out(x,y), u_range=[-2.7, 2.7], v_range=[-2.7, 2.7],
                           resolution=50, stroke_width=0, fill_opacity=0.9)
        ham_surf.move_to(4*RIGHT).scale(0.7)
        ham_surf.set_fill_by_value(axes=axes, colorscale=[(ManimColor("#456354"),-1), (ManimColor("#ABF5D1"),1)])



        self.play(Create(axis))
        self.play(*[GrowArrow(i) for i in VF_HNN])
        self.play(Create(p_dot))
        self.add(p_dot_trace)
        self.wait(1)
        self.move_camera(phi=0.3*PI, theta=0.3*PI, frame_center=[4,0,1], run_time=2, added_anims=[Transform(VF_HNN, ham_surf)])
        self.move_camera(phi=0.35*PI, theta=2.15*PI, run_time=4)
        for i in range(1, 2):
            x, y = 0.7 * q_HNN[2 * i] + 4, 0.7 * p_HNN[2 * i]
            self.play(p_dot.animate.move_to([x, y, 0]), run_time=0.01)


class all(Slide):
    def construct(self):
        test1.construct(self)
        test2.construct(self)
        test3.construct(self)