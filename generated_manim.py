from manim import *

class TypesOfNumbers(Scene):
    def construct(self):
        # Title
        title = Text("Types of Numbers", font_size=42).to_edge(UP)
        self.play(Write(title))

        # Circles for different sets
        real_circle = Circle(radius=3.5, color=YELLOW).shift(LEFT)
        rational_circle = Circle(radius=2.6, color=BLUE).shift(LEFT*1.4 + DOWN*0.8)
        irrational_circle = Circle(radius=2.6, color=GREEN).shift(RIGHT*1.2 + DOWN*0.7)
        integer_circle = Circle(radius=1.5, color=RED).shift(LEFT*2)
        whole_circle = Circle(radius=0.8, color=PURPLE).shift(LEFT*2.5 + UP*0.5)
        natural_circle = Circle(radius=0.45, color=ORANGE).shift(LEFT*2.5 + UP*1.15)

        # Labels
        real_label = Text("Real Numbers", font_size=26, color=YELLOW).next_to(real_circle, UP, buff=0.1)
        rational_label = Text("Rational Numbers", font_size=22, color=BLUE).next_to(rational_circle, DOWN, buff=0.1)
        irrational_label = Text("Irrational Numbers", font_size=22, color=GREEN).next_to(irrational_circle, DOWN, buff=0.1)
        integer_label = Text("Integers", font_size=22, color=RED).next_to(integer_circle, LEFT, buff=0.05)
        whole_label = Text("Whole Numbers", font_size=18, color=PURPLE).next_to(whole_circle, UP, buff=0.05)
        natural_label = Text("Natural Numbers", font_size=16, color=ORANGE).next_to(natural_circle, UP, buff=0.05)

        # Complex Numbers Around
        complex_rect = Rectangle(width=10, height=6.5, color=TEAL).shift(DOWN*0.7)
        complex_label = Text("Complex Numbers", font_size=28, color=TEAL).next_to(complex_rect, UP, buff=0.15)

        # Number Examples
        examples = VGroup(
            Tex(r"\\mathbb{N}: \\ \{1, 2, 3,...\}", color=ORANGE).scale(0.65).next_to(natural_circle, UP*2, buff=0.15),
            Tex(r"\\mathbb{W}: \\ \{0, 1, 2,...\}", color=PURPLE).scale(0.65).next_to(whole_circle, LEFT*1.5, buff=0.05),
            Tex(r"\\mathbb{Z}: ...,-2,-1,0,1,2,...", color=RED).scale(0.7).next_to(integer_circle, LEFT*2.2, buff=0.05),
            Tex(r"\\mathbb{Q}: 1/2, -3/4, 0.25", color=BLUE).scale(0.7).next_to(rational_circle, LEFT*1.7 + DOWN*0.85, buff=0.05),
            Tex(r"I: \\sqrt{2}, \\pi", color=GREEN).scale(0.7).next_to(irrational_circle, RIGHT*1.6 + DOWN*1, buff=0.05),
            Tex(r"\\mathbb{R}: \\mathbb{Q} \\cup I", color=YELLOW).scale(0.7).next_to(real_circle, DOWN*2, buff=0.01),
            Tex(r"\\mathbb{C}: 2+3i", color=TEAL).scale(0.7).next_to(complex_rect, DOWN*3.3, buff=0.05),
        )

        # Draw everything
        self.play(
            DrawBorderThenFill(complex_rect),
            Write(complex_label)
        )
        self.play(Create(real_circle), Write(real_label))
        self.play(Create(rational_circle), Write(rational_label))
        self.play(Create(irrational_circle), Write(irrational_label))
        self.play(Create(integer_circle), Write(integer_label))
        self.play(Create(whole_circle), Write(whole_label))
        self.play(Create(natural_circle), Write(natural_label))
        self.play(*[Write(ex) for ex in examples])
        self.wait(2)
