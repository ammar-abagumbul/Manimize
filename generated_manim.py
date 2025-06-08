from manim import *

class TypesOfTriangles(Scene):
    def construct(self):
        title = Text("Types of Triangles", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))

        # By Sides
        sides_title = Text("By Sides", font_size=28).to_edge(LEFT).shift(DOWN*0.5)
        self.play(Write(sides_title))

        # Scalene triangle
        scalene = Polygon(
            [0, 0, 0],
            [2, 0, 0],
            [0.5, 1.5, 0],
            color=BLUE
        ).shift(LEFT*4 + DOWN*2)
        scalene_label = Text("Scalene", font_size=24).next_to(scalene, DOWN)
        # Isosceles triangle
        isosceles = Polygon(
            [0, 0, 0],
            [2, 0, 0],
            [1, 1.5, 0],
            color=GREEN
        ).shift(LEFT*0.5 + DOWN*2)
        isosceles_label = Text("Isosceles", font_size=24).next_to(isosceles, DOWN)
        # Equilateral triangle
        from numpy import array
        h = (3**0.5)/2*2
        equilateral = Polygon(
            [0, 0, 0],
            [2, 0, 0],
            [1, h, 0],
            color=YELLOW
        ).shift(RIGHT*3 + DOWN*2)
        equilateral_label = Text("Equilateral", font_size=24).next_to(equilateral, DOWN)
        self.play(Create(scalene), Write(scalene_label))
        self.play(Create(isosceles), Write(isosceles_label))
        self.play(Create(equilateral), Write(equilateral_label))

        # By Angles
        angles_title = Text("By Angles", font_size=28).to_edge(LEFT).shift(UP*0.5)
        self.play(FadeOut(scalene), FadeOut(isosceles), FadeOut(equilateral),
                  FadeOut(scalene_label), FadeOut(isosceles_label), FadeOut(equilateral_label),
                  FadeOut(sides_title))
        self.play(Write(angles_title))

        # Acute triangle (all angles < 90)
        acute = Polygon(
            [0, 0, 0],
            [2, 0, 0],
            [1, 1.3, 0],
            color=PURPLE
        ).shift(LEFT*4 + UP*1)
        acute_label = Text("Acute", font_size=24).next_to(acute, DOWN)
        # Right triangle (one angle = 90)
        right = Polygon(
            [0, 0, 0],
            [2, 0, 0],
            [0, 1.5, 0],
            color=RED
        ).shift(LEFT*0.5 + UP*1)
        right_label = Text("Right", font_size=24).next_to(right, DOWN)
        # Small square for right angle
        right_angle = (
            Square(0.3, color=WHITE, fill_opacity=1)
            .move_to([0.2, 0.2, 0])
            .shift(LEFT * 0.5 + UP * 1)
        )
        # Obtuse triangle (one angle > 90)
        obtuse = Polygon([0, 0, 0], [2, 0, 0], [1.7, 0.8, 0], color=ORANGE).shift(
            RIGHT * 3 + UP * 1
        )
        obtuse_label = Text("Obtuse", font_size=24).next_to(obtuse, DOWN)
        self.play(Create(acute), Write(acute_label))
        self.play(Create(right), Write(right_label), FadeIn(right_angle))
        self.play(Create(obtuse), Write(obtuse_label))
        self.wait(2)

