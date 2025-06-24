from manim import *

class RiemannHypothesis(Scene):
    def construct(self):
        # Introduction to Riemann Hypothesis
        title = Text("The Riemann Hypothesis", font_size=48)
        intro_text = Text(
            "Proposed by Bernhard Riemann in 1859",
            font_size=24
        ).next_to(title, DOWN, buff=0.5)
        hypothesis = MathTex(
            r"\text{All non-trivial zeros of the zeta function } \zeta(s) \text{ have real part } \frac{1}{2}",
            font_size=36
        ).next_to(intro_text, DOWN, buff=1.0)

        self.play(Write(title))
        self.play(Write(intro_text))
        self.play(Write(hypothesis))
        self.wait(2)

        # Transition to visualization of the zeta function
        self.play(FadeOut(title), FadeOut(intro_text), FadeOut(hypothesis))

        # Visualize the complex plane and critical strip
        plane = ComplexPlane(
            x_range=(-2, 2, 0.5),
            y_range=(-2, 2, 0.5),
            background_line_style={"stroke_opacity": 0.3}
        ).add_coordinates()
        plane_label = Text("Complex Plane", font_size=24).to_edge(UP)
        critical_strip = VGroup(
            Line(start=plane.c2p(0, -2), end=plane.c2p(0, 2), color=YELLOW),
            Line(start=plane.c2p(1, -2), end=plane.c2p(1, 2), color=YELLOW),
        )
        critical_line = Line(start=plane.c2p(0.5, -2), end=plane.c2p(0.5, 2), color=RED)
        strip_label = Text("Critical Strip", font_size=18, color=YELLOW).next_to(plane.c2p(0.5, 1.5), RIGHT)
        line_label = Text("Critical Line (Re=1/2)", font_size=18, color=RED).next_to(plane.c2p(0.5, 1.2), RIGHT)

        self.play(Create(plane), Write(plane_label))
        self.play(Create(critical_strip), Write(strip_label))
        self.play(Create(critical_line), Write(line_label))
        self.wait(2)

        # Explain the zeros of the zeta function
        zero_dots = VGroup(
            Dot(plane.c2p(0.5, 0.5), color=BLUE),
            Dot(plane.c2p(0.5, -0.5), color=BLUE),
            Dot(plane.c2p(0.5, 1.0), color=BLUE),
            Dot(plane.c2p(0.5, -1.0), color=BLUE),
        )
        zero_label = Text("Non-trivial Zeros", font_size=18, color=BLUE).next_to(plane.c2p(0.5, -1.5), RIGHT)
        prime_text = Text(
            "Zeros relate to the distribution of prime numbers",
            font_size=24
        ).to_edge(DOWN)

        self.play(Create(zero_dots), Write(zero_label))
        self.play(Write(prime_text))
        self.wait(2)

        # Transition to implications
        self.play(FadeOut(plane), FadeOut(critical_strip), FadeOut(critical_line),
                  FadeOut(strip_label), FadeOut(line_label), FadeOut(zero_dots),
                  FadeOut(zero_label), FadeOut(prime_text), FadeOut(plane_label))

        # Implications and importance
        imp_title = Text("Why It Matters", font_size=36)
        imp_text1 = Text(
            "Key to understanding prime number distribution",
            font_size=24
        ).next_to(imp_title, DOWN, buff=0.5)
        imp_text2 = Text(
            "A Millennium Prize Problem ($1M prize)",
            font_size=24
        ).next_to(imp_text1, DOWN, buff=0.5)

        self.play(Write(imp_title))
        self.play(Write(imp_text1))
        self.play(Write(imp_text2))
        self.wait(3)

        # End scene
        self.play(FadeOut(imp_title), FadeOut(imp_text1), FadeOut(imp_text2))
        end_text = Text("The Riemann Hypothesis remains unsolved", font_size=36)
        self.play(Write(end_text))
        self.wait(2)
        self.play(FadeOut(end_text))
