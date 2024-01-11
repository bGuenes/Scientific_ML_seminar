from manim import *
from manim_slides import Slide
import autograd.numpy as np
import autograd
from classes import *
import torch
import torch.nn as nn

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# setup
background_color = "#ECE7E2"
font = "Helvetica"
tex_font = TexFontTemplates.helvetica_fourier_it


def hamiltonian(x):
    q, p = np.split(x, 2)
    return 3 * (1 - autograd.numpy.cos(q)) + p ** 2

# Load data
q = np.load("../data/pendulum/x.npy")
p = np.load("../data/pendulum/y.npy")

q_NN, p_NN = np.load("../data/pendulum/xNN.npy")


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 1
class S1(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = background_color

        # Create texts
        title = Text("Hamiltonian Neural Networks", font=font, font_size=65, color=BLACK, weight=BOLD).move_to(
            UP)
        subtitle = Text("Seminar: Scientific Machine Learning", font=font, font_size=30, color=BLACK).next_to(
            title, DOWN)
        footer = Text("Berkay Günes", font=font, font_size=12, color=BLACK).to_edge(DL)
        date = Text("15.01.2024", font=font, font_size=12, color=BLACK).to_edge(DR)

        # -----------------------------------
        # Animate slide 1
        self.next_slide()
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.play(FadeIn(footer), FadeIn(date))
        self.next_slide()
        self.clear()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 2
class S2(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = background_color
        # Create title
        title = Text("Pendulum", font=font, font_size=60, color=BLACK)

        # Create pendulum components
        l = -3
        start = [0, 0, 0]
        end = [0, l, 0]

        pivot = Dot(start, color=BLACK)
        rod = Line(start, end, color=BLACK)
        bob = Dot(end, color="#94424F", radius=0.2)

        pendulum = VGroup(pivot, rod, bob)

        # Crete phase space

        func = lambda x: autograd.grad(hamiltonian)(np.array([x[0], x[1]])) @ np.array([[0,-1], [1,0]])

        axis = Axes(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1], axis_config={"color": BLACK}, x_length=7, y_length=7)
        labels = axis.get_axis_labels(Tex("q", color=BLACK).scale(0.5), Tex("p", color=BLACK).scale(0.5))
        axis = VGroup(axis, labels)
        VF = ArrowVectorField(func, x_range=[-2.5, 2.5, 0.4], y_range=[-2.5, 2.5, 0.4], colors=["#ABF5D1", "#87C2A5", "#456354"])

        field = VGroup(axis, VF).shift(4*RIGHT).scale(0.7)

        x0, y0 = 0.6 * q[0] + 4, 0.6 * p[0]
        p_dot = Dot([x0, y0, 0], color="#94424F", radius=0.05)
        p_dot_trace = TracedPath(p_dot.get_center, stroke_color="#94424F")

        # -----------------------------------
        # Animations slide 2
        self.play(Write(title))

        self.play(title.animate.to_corner(UL))
        self.play(Create(rod), Create(pivot), Create(bob))
        self.wait(1)

        self.play(pendulum.animate.shift(3.5*LEFT))
        self.play(Create(axis))
        self.play(*[GrowArrow(i) for i in VF])

        self.next_slide()
        self.play(Rotate(pendulum, q[0], about_point=[-3.5, 0, 0]), run_time=1)
        self.play(Create(p_dot))
        self.add(p_dot_trace)

        # Animate pendulum using existing data
        self.next_slide(loop=True)
        for i in range(1, 100):
            angle = q[2*i] - q[2*(i-1)]
            self.play(Rotate(pendulum, angle, about_point=[-3.5, 0, 0]), run_time=0.01)

            x, y = 0.6 * q[2*i] + 4, 0.6 * p[2*i]
            self.play(p_dot.animate.move_to([x, y, 0]), run_time=0.01)
        self.next_slide()
        self.clear()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 3
class S3(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = background_color

        title = Text("Baseline NN", font=font, font_size=60, color=BLACK).to_edge(UL)

        # Create NN
        NN = NeuralNetworkMobject([2, 5, 2]).scale(1.3)
        NN.label_inputs(["q", "p"])
        NN.label_outputs(["dq", "dp"])

        CArrow1 = CurvedArrow([2, -2, 0], [2, 1, 0], angle=3, color=BLACK)
        text1 = Text("integrate", font=font, color=BLACK).move_to(UP)
        CArrow2 = CurvedArrow([-2, 1, 0], [-2, -2, 0], angle=3, color=BLACK)

        NN_circle = VGroup(NN, CArrow1, text1, CArrow2)

        # Crete phase space
        axis = Axes(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1], axis_config={"color": BLACK}, x_length=7, y_length=7)
        labels = axis.get_axis_labels(Tex("q", color=BLACK).scale(0.5), Tex("p", color=BLACK).scale(0.5))
        axis = VGroup(axis, labels)

        NN_model = MLP()
        NN_model.load_state_dict(torch.load("../data/pendulum/model_NN.pt"))
        NN_model.eval()

        def func(x):
            x = torch.tensor(x[0:2], dtype=torch.float32)
            return NN_model(x).detach().numpy()

        VF_NN = ArrowVectorField(func, x_range=[-2.5, 2.5, 0.4], y_range=[-2.5, 2.5, 0.4], colors=["#ABF5D1", "#87C2A5", "#456354"])
        field_NN = VGroup(axis, VF_NN).shift(4*RIGHT).scale(0.7)

        x0, y0 = 0.6 * q_NN[0] + 4, 0.6 * p_NN[0]
        p_dot = Dot([x0, y0, 0], color="#94424F", radius=0.05)
        p_dot_trace = TracedPath(p_dot.get_center, stroke_color="#94424F")

        # Create pendulum
        l = -3
        start = [0, 0, 0]
        end = [0, l, 0]

        pivot = Dot(start, color=BLACK)
        rod = Line(start, end, color=BLACK)
        bob = Dot(end, color="#94424F", radius=0.2)

        pendulum = VGroup(pivot, rod, bob).shift(3.5*LEFT)

        # -----------------------------------
        # Animate slide 3

        self.play(Write(title), run_time=1)
        self.wait(1)
        self.play(Create(NN), run_time=3)

        self.next_slide()
        self.play(NN.animate.move_to(2 * DOWN))
        self.play(Create(CArrow1), run_time=1)
        self.play(Write(text1), run_time=1)

        self.next_slide()
        self.play(Create(CArrow2), run_time=1)

        self.next_slide()
        self.play(NN_circle.animate.scale(0.2).next_to(title, RIGHT, buff=1), run_time=2)
        self.play(Create(pendulum))
        self.play(Create(axis))
        self.play(*[GrowArrow(i) for i in VF_NN])

        self.next_slide()
        self.play(Rotate(pendulum, q_NN[0], about_point=[-3.5, 0, 0]), run_time=1)
        self.play(Create(p_dot))
        self.add(p_dot_trace)

        self.next_slide()
        for i in range(1, 600):
            angle = q_NN[2*i] - q_NN[2*(i-1)]
            self.play(Rotate(pendulum, angle, about_point=[-3.5, 0, 0]), run_time=0.01)

            x, y = 0.6 * q_NN[2*i] + 4, 0.6 * p_NN[2*i]
            self.play(p_dot.animate.move_to([x, y, 0]), run_time=0.01)

        # -----------------------------------
        # Animate slide 3.5
        fade_all = VGroup(NN_circle, pendulum)

        func = lambda x: autograd.grad(hamiltonian)(np.array([x[0], x[1]])) @ np.array([[0,-1], [1,0]])

        axis2 = Axes(x_range=[-3.5, 3.5, 1], y_range=[-3.5, 3.5, 1], axis_config={"color": BLACK}, x_length=7, y_length=7)
        labels2 = axis2.get_axis_labels(Tex("q", color=BLACK).scale(0.5), Tex("p", color=BLACK).scale(0.5))
        axis2 = VGroup(axis2, labels2)
        VF2 = ArrowVectorField(func, x_range=[-2.5, 2.5, 0.4], y_range=[-2.5, 2.5, 0.4], colors=["#ABF5D1", "#87C2A5", "#456354"])

        field = VGroup(axis2, VF2).shift(4*LEFT).scale(0.7)

        title_GT = Text("Reality", font=font, font_size=30, color=BLACK).next_to(field, UP, buff=0.1)
        #title_NN = Text("Baseline NN", font=font, font_size=30, color=BLACK).next_to(field_NN, DOWN, buff=0.2)
        #grap_subs = VGroup(title_GT, title_NN)

        x0, y0 = 0.6 * q[0] - 4, 0.6 * p[0]
        p_dot2 = Dot([x0, y0, 0], color="#94424F", radius=0.05)
        p_dot_trace2 = TracedPath(p_dot2.get_center, stroke_color="#94424F")

        message1 = Text("Conserving energy!", font=font, font_size=20, color=BLACK).next_to(field, DOWN, buff=0.5)
        message = Text("Spiraling out of control!", font=font, font_size=20, color=BLACK).next_to(field_NN, DOWN, buff=0.5)

        self.next_slide()
        self.play(FadeOut(fade_all))
        self.play(title.animate.next_to(field_NN, UP, buff=0.1).scale(0.5))
        self.play(Create(axis2))
        self.play(*[GrowArrow(i) for i in VF2])
        self.play(Write(title_GT))

        self.play(Create(p_dot2))
        self.add(p_dot_trace2)
        for i in range(1, 200):
            x, y = 0.6 * q[2*i] - 4, 0.6 * p[2*i]
            self.play(p_dot2.animate.move_to([x, y, 0]), run_time=0.005)


        self.next_slide()
        self.play(Write(message1))
        self.next_slide()
        self.play(Write(message))
        self.next_slide()
        self.clear()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 4
class S4(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = background_color

        title = Text("Hamiltonian physics", font=font, font_size=60, color=BLACK)
        title2 = Text("Recap Hamiltonian", font=font, font_size=60, color=BLACK).to_edge(UL)

        texts = VGroup()

        p1 = Tex(r"$\bullet$ Set of $\mathbf{(q,p)}$ describes full system"
                 , tex_template=tex_font, font_size=45, color=BLACK)
        texts.add(p1)

        p2 = Tex(r"$\bullet$ $\mathcal H$, so that "
                 r"$\boldsymbol{\dot q} = \frac{\partial \mathcal H}{\partial \boldsymbol{p}}$"
                 r" and "
                 r"$\boldsymbol{\dot p} = -\frac{\partial \mathcal H}{\partial \boldsymbol{q}}$"
                 , tex_template=tex_font, font_size=45, color=BLACK)
        texts.add(p2)

        p3 = Tex(r"$\bullet$ $\mathcal H = E_{tot} = T + V$", tex_template=tex_font, font_size=45, color=BLACK)
        texts.add(p3)

        p4 = Tex(r"$\bullet$ Symplectic gradient "
                 r"$\boldsymbol{S}_{\mathcal H} = (\frac{\partial \mathcal H}{\partial \boldsymbol{p}},-\frac{\partial \mathcal H}{\partial \boldsymbol{q}})$"
                 , tex_template=tex_font, font_size=45, color=BLACK)
        texts.add(p4)

        p5 = Tex(r"$\bullet$ Describe time evolution with:"
                 , tex_template=tex_font, font_size=45, color=BLACK)
        texts.add(p5)

        p6 = Tex(r"     $(q_{n+1},p_{n+1}) = (q_n,p_n) + {\Huge \int} \boldsymbol{S}_{\mathcal H}(\boldsymbol{q},\boldsymbol{p})$"
                 , tex_template=tex_font, font_size=45, color=BLACK)
        texts.add(p6)

        texts.arrange(1.5*DOWN, center=False, aligned_edge=LEFT).next_to(title2, DOWN, buff=1).align_to(title2, LEFT).shift(RIGHT)

        # -----------------------------------
        # Animate slide 4
        self.play(Write(title))
        self.next_slide()
        self.play(Transform(title, title2))
        self.next_slide()
        self.play(Write(p1))
        self.next_slide()
        self.play(Write(p2))
        self.next_slide()
        self.play(Write(p3))
        self.next_slide()
        self.play(Write(p4))
        self.next_slide()
        self.play(Write(p5))
        self.play(Write(p6.shift(RIGHT+0.1*UP)))
        self.next_slide()

        self.clear()

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 5
class S5(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = background_color

        dHq = r"\frac{\partial \mathcal H}{\partial \boldsymbol{q}}"
        dHp = r"\frac{\partial \mathcal H}{\partial \boldsymbol{p}}"
        dq = r"\boldsymbol{\dot q}"
        dp = r"\boldsymbol{\dot p}"

        eq1 = MathTex(dq, "=", dHp
                      , tex_template=tex_font, font_size=60, color=BLACK).move_to(2*LEFT)
        eq2 = MathTex(dp, "=", "-", dHq
                      , tex_template=tex_font, font_size=60, color=BLACK).move_to(2*RIGHT)

        eq12 = MathTex(dq, "-", dHp, "=", "0"
                      , tex_template=tex_font, font_size=60, color=BLACK).move_to(2*LEFT)
        eq22 = MathTex(dp, "+", dHq, "=", "0"
                      , tex_template=tex_font, font_size=60, color=BLACK).move_to(2 * RIGHT)

        text1 = Text("Minimize energy loss", font=font, font_size=45, color=BLACK).next_to(eq1, UP, buff=1)

        eq3 = MathTex("0", "=", "(", dq, "-", dHp, ")^2", "+", "(", dp, "+", dHq, ")^2"
                      , tex_template=tex_font, font_size=60, color=BLACK)

        brace_L2 = Brace(eq3[2:-1], DOWN, buff=0.3, color=BLACK)
        brace_L2_text = Text("L2 Loss", font=font, font_size=30, color=BLACK).next_to(brace_L2, DOWN, buff=0.1)

        eq4 = MathTex(r"\text{Loss}", "=", r"\text{argmin}_{\mathcal H}", "(", dq, "-", dHp, ")^2", "+", "(", dp, "+", dHq, ")^2"
                      , tex_template=tex_font, font_size=60, color=BLACK)

        text2 = Text("Not optimizing the network output, but its gradient!"
                     , font=font, font_size=40, color=BLACK, t2w={"optimizing":HEAVY, "gradient":HEAVY}).move_to(2*DOWN)


        # -----------------------------------
        # Animate slide 5
        self.play(Write(eq1), Write(eq2))
        self.next_slide()
        self.play(FadeIn(text1))
        self.next_slide()
        self.play(TransformMatchingTex(eq1, eq12), TransformMatchingTex(eq2, eq22))
        self.next_slide()
        self.play(TransformMatchingTex(Group(eq12, eq22), eq3))
        self.next_slide()
        self.play(GrowFromCenter(brace_L2), FadeIn(brace_L2_text))
        self.next_slide()
        self.play(FadeOut(brace_L2), FadeOut(brace_L2_text))
        self.play(TransformMatchingTex(eq3, eq4))
        self.next_slide()
        self.play(Write(text2))

        self.clear()


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Slide 6
class S6(Slide):
    def construct(self):
        # Set background color
        self.camera.background_color = background_color

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////
# play all slides
class all(Slide):
    def construct(self):
        S1.construct(self)
        S2.construct(self)
        S3.construct(self)
        S4.construct(self)
        S5.construct(self)
