from manim import *


class RationalNumbersScene(Scene):
    def construct(self):
        # Title
        title = Text("Rational Numbers", font_size=48).to_edge(UP)
        underline = Line(title.get_left(), title.get_right()).next_to(
            title, DOWN, buff=0.1
        )
        self.play(Write(title), Create(underline))
        self.wait(0.5)

        # Definition textbox
        definition = (
            "A rational number can be written as a fraction $\\dfrac{a}{b}$"
            "where $a, b \\in \\mathbb{Z}$ and $b \\neq 0$."
        )

        def_box = (
            Rectangle(stroke_color=BLUE, height=2, width=6)
            .to_edge(LEFT)
            .shift(DOWN * 0.5)
        )
        def_text = Tex(definition, font_size=36).move_to(def_box.get_center())
        self.play(FadeIn(def_box), Write(def_text, run_time=3))
        self.wait(1)

        # Symbol
        q_symbol = (
            MathTex("\\mathbb{Q}", font_size=60, color=YELLOW)
            .to_edge(RIGHT)
            .shift(UP * 1)
        )
        label = Text("The set of rational numbers", font_size=28).next_to(
            q_symbol, DOWN, buff=0.2
        )
        self.play(Write(q_symbol), Write(label))
        self.wait(0.7)

        # Examples
        ex_title = (
            Text("Examples:", font_size=36)
            .next_to(def_box, DOWN, buff=0.7)
            .align_to(def_box, LEFT)
        )
        examples = (
            VGroup(
                MathTex("\\frac{1}{2}", font_size=36),
                MathTex("-\\frac{3}{7}", font_size=36),
                MathTex("5 = \\frac{5}{1}", font_size=36),
                MathTex("0 = \\frac{0}{8}", font_size=36),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            .next_to(ex_title, DOWN, buff=0.15)
            .align_to(ex_title, LEFT)
        )
        self.play(Write(ex_title))
        self.play(LaggedStart(*[FadeIn(e) for e in examples], lag_ratio=0.2))
        self.wait(1)

        # Abelian group under addition
        add_group_title = Text(
            "Abelian Group under Addition", font_size=32, color=GREEN
        ).shift(DOWN * 0.2 + RIGHT * 2.1)
        add_props = (
            BulletedList(
                "$\\forall~x,y \\in \\mathbb{Q}: x + y \\in \\mathbb{Q}$ (Closed)",
                "Identity: $0$",
                "Inverse: $\\forall~x,\\ x + (-x) = 0$",
                "Associative: $(x+y)+z = x+(y+z)$",
                "Commutative: $x+y = y+x$",
                font_size=26,
                dot_scale_factor=1.1,
            )
            .next_to(add_group_title, DOWN, aligned_edge=LEFT, buff=0.2)
            .shift(RIGHT * 2.1)
        )
        add_example = MathTex(
            "e.g.,\quad \\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}", font_size=28
        ).next_to(add_props, DOWN)
        self.play(Write(add_group_title))
        self.play(FadeIn(add_props, shift=RIGHT))
        self.wait(0.7)
        self.play(Write(add_example))
        self.wait(1)

        # Abelian group under multiplication (excluding 0)
        mul_group_title = Text(
            "Abelian Group under Multiplication ($\\mathbb{Q}^*$)",
            font_size=32,
            color=PURPLE,
        ).next_to(add_example, DOWN, buff=0.5, aligned_edge=LEFT)
        mul_props = BulletedList(
            "$\\forall~x,y \\in \\mathbb{Q}^*: x \\cdot y \\in \\mathbb{Q}^*$",
            "Identity: $1$",
            "Inverse: $\\forall~x \\neq 0,\\ x \\cdot \\frac{1}{x} = 1$",
            "Associative: $(xy)z = x(yz)$",
            "Commutative: $xy = yx$",
            font_size=26,
            dot_scale_factor=1.1,
        ).next_to(mul_group_title, DOWN, aligned_edge=LEFT, buff=0.2)
        mul_example = MathTex(
            "e.g.,\quad \\frac{2}{3} \\times \\frac{9}{8} = \\frac{3}{4}", font_size=28
        ).next_to(mul_props, DOWN)
        self.play(Write(mul_group_title))
        self.play(FadeIn(mul_props, shift=RIGHT))
        self.wait(0.7)
        self.play(Write(mul_example))
        self.wait(1.3)

        # Highlight Q wrap-up
        q_brace = Brace(q_symbol, LEFT, color=YELLOW)
        q_label = q_brace.get_text("Rational numbers").set_color(YELLOW)
        self.play(GrowFromCenter(q_brace), Write(q_label))
        self.wait(1)

        # Fade everything but Q and the definition for empahsis
        to_fade = VGroup(
            add_group_title,
            add_props,
            add_example,
            mul_group_title,
            mul_props,
            mul_example,
            ex_title,
            examples,
            label,
        )
        self.play(FadeOut(to_fade, shift=DOWN), rate_func=smooth)
        self.wait(0.5)
        self.play(
            q_symbol.animate.move_to(ORIGIN + UP * 1.1).scale(1.3),
            q_label.animate.next_to(q_symbol, DOWN),
            FadeOut(def_box),
            def_text.animate.move_to(ORIGIN + DOWN * 1.2),
        )

        # End
        self.wait(2)
