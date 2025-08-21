from manim import *
import numpy as np
import random
import itertools

class CentralLimitTheoremScene(Scene):
    """
    A complete animated exposition of the Central Limit Theorem
    following the outline provided.
    """

    # ----------------------------------------------------------------------
    # Helper utilities -----------------------------------------------------
    # ----------------------------------------------------------------------
    def dice_rolls(self, n, dice=6):
        """Return a list of n dice rolls (1‑dice)."""
        return [random.randint(1, dice) for _ in range(n)]

    def histogram(self, data, bins, **kwargs):
        """
        Return a BarChart representing the histogram of *data*.
        ``bins`` can be an int (number of equal‑width bins) or an
        array of bin edges.
        """
        if isinstance(bins, int):
            bin_edges = np.linspace(min(data) - 0.5,
                                    max(data) + 0.5,
                                    bins + 1)
        else:
            bin_edges = np.array(bins)

        counts, _ = np.histogram(data, bins=bin_edges)
        # Normalise for nicer animation size
        max_count = max(counts) if max(counts) != 0 else 1
        heights = [c / max_count for c in counts]

        bars = BarChart(
            values=heights,
            bar_names=[f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
                       for i in range(len(heights))],
            y_range=[0, 1.2, 0.2],
            x_length=8,
            y_length=4,
            bar_fill_opacity=0.8,
            **kwargs,
        )
        return bars

    def normal_curve(self, mu=0, sigma=1, length=8):
        """Return a parametric curve of the Gaussian pdf."""
        # Scale to match histogram width (8 units)
        def pdf(x):
            return (1/(sigma*np.sqrt(2*np.pi)) *
                    np.exp(-0.5*((x-mu)/sigma)**2))

        # Sample points for a smooth curve
        xs = np.linspace(-4*sigma, 4*sigma, 200)
        pts = np.column_stack([xs, pdf(xs) * length/ (np.max(pdf(xs))) ])
        curve = VMobject()
        curve.set_points_smoothly([np.array([p[0], p[1], 0]) for p in pts])
        curve.set_stroke(RED, width=4)
        return curve

    def convolution_step(self, dist1, dist2, x_range=(-5,5), points=400):
        """Numerically convolve two pdf arrays (given as callable)."""
        xs = np.linspace(*x_range, points)
        conv = np.convolve([dist1(x) for x in xs],
                           [dist2(x) for x in xs],
                           mode='same')
        conv = conv / np.max(conv)    # normalise
        pts = np.column_stack([xs, conv])
        curve = VMobject().set_stroke(BLUE, width=3)
        curve.set_points_smoothly([np.array([p[0], p[1], 0]) for p in pts])
        return curve

    # ----------------------------------------------------------------------
    # Section 1 – Intuitive Hook: Rolling Dice -----------------------------
    # ----------------------------------------------------------------------
    def dice_section(self):
        title = Text("Intuitive Hook – Rolling Dice", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=UP))

        # Show 5 dice faces rolling
        dice_sprites = VGroup()
        for i in range(5):
            # a simple square with a number inside
            face = Square(side_length=0.8, fill_opacity=0.8,
                          fill_color=BLUE_E).shift(2*LEFT + i*RIGHT)
            txt = Integer(random.randint(1,6), font_size=48).move_to(face)
            dice_sprites.add(VGroup(face, txt))
        self.play(FadeIn(dice_sprites))

        # Simulate many throws and accumulate into a histogram
        n_throws = 200
        throws = self.dice_rolls(n_throws)
        hist = self.histogram(throws, bins=6, bar_names=[str(i) for i in range(1,7)])
        hist.next_to(title, DOWN, buff=0.8)
        self.play(Create(hist))

        # Morph histogram into a smooth Gaussian curve
        gauss = self.normal_curve(mu=3.5, sigma=1.5, length=4)
        gauss.next_to(hist, DOWN, buff=0.5)

        # Fade out squares, keep bars, then transform
        self.play(FadeOut(dice_sprites, shift=DOWN), run_time=1)
        self.play(Transform(hist, gauss), run_time=2)
        self.wait(1)

        self.play(FadeOut(VGroup(title, hist, gauss), shift=DOWN))
        self.wait(0.5)

    # ----------------------------------------------------------------------
    # Section 2 – Formal Statement -----------------------------------------
    # ----------------------------------------------------------------------
    def formal_statement_section(self):
        title = Text("Formal Statement of the CLT", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=UP))

        # Plain‑language statement
        statement = Tex(
            r“For a large enough sample size $n$, the distribution of the sample "
            r"mean $\\bar X = \\frac{1}{n}\\sum_{i=1}^{n}X_i$ of i.i.d. random variables "
            r"with finite variance tends to a normal distribution $\\mathcal N(\\mu,\\sigma^2/n)$.",
            font_size=28,
            tex_environment="flushleft"
        ).next_to(title, DOWN, buff=0.7)
        self.play(Write(statement))
        self.wait(2)

        # Diagram of boxes → mean → Gaussian
        boxes = VGroup(*[Square(side_length=0.6, fill_opacity=0.6,
                               fill_color=PURPLE).shift(LEFT*2 + i*RIGHT*0.8)
                        for i in range(5)])
        arrows = VGroup(*[Arrow(start=box.get_right(),
                                end=box.get_right()+0.6*RIGHT,
                                buff=0)
                          for box in boxes])
        mean_calc = Rectangle(width=1.6, height=0.8,
                              fill_opacity=0.5,
                              fill_color=YELLOW).next_to(arrows, RIGHT, buff=0)
        mean_label = Tex(r"$\\bar X$", font_size=36).move_to(mean_calc)

        # Arrow to Gaussian silhouette
        to_gauss = Arrow(start=mean_calc.get_right(),
                         end=mean_calc.get_right()+2*RIGHT,
                         buff=0)
        gauss_sil = self.normal_curve(mu=0, sigma=1, length=3)
        gauss_sil.move_to(to_gauss.get_end())

        diagram = VGroup(boxes, arrows, mean_calc, mean_label,
                         to_gauss, gauss_sil)
        diagram.to_edge(DOWN, buff=1)

        self.play(FadeIn(diagram, scale=0.5))
        self.wait(2)

        # Show convergence of three different original distributions
        dist_names = ["Uniform", "Exponential", "Bernoulli"]
        colors = [GREEN, ORANGE, RED]
        curves = VGroup()
        for i, name in enumerate(dist_names):
            # simple pdfs
            if name == "Uniform":
                func = lambda x: 1/6 if 1 <= x <= 6 else 0
                xs = np.linspace(0, 7, 300)
            elif name == "Exponential":
                func = lambda x: np.exp(-x) if x >= 0 else 0
                xs = np.linspace(0, 7, 300)
            else:  # Bernoulli {0,1}
                func = lambda x: 0.5 if x in (0,1) else 0
                xs = np.linspace(-0.5, 1.5, 200)

            ys = [func(x) for x in xs]
            curve = VMobject(stroke_color=colors[i], stroke_width=2)
            points = np.column_stack([xs, ys])
            curve.set_points_smoothly([np.array([p[0], p[1], 0]) for p in points])
            curve.shift(4*LEFT + i*2*RIGHT)
            curves.add(curve)

            label = Tex(name, font_size=24).next_to(curve, DOWN)
            curves.add(label)

        self.play(FadeIn(curves, shift=UP))
        self.wait(1)

        # Animate increase of sample size by convolving the same pdf with itself
        # (very simplified visual)
        conv_label = Tex(r"Sample size $n$ grows → repeated convolution", font_size=24)
        conv_label.next_to(curves, UP)
        self.play(Write(conv_label))
        self.wait(1)

        # Replace each curve by a Gaussian (using Transform)
        for i, curve in enumerate(curves):
            if not isinstance(curve, VMobject):
                continue
            gauss = self.normal_curve(mu=0, sigma=1, length=2)
            gauss.move_to(curve)
            self.play(Transform(curve, gauss), run_time=1.5)

        self.wait(1)
        self.play(FadeOut(VGroup(title, statement, diagram, curves, conv_label), shift=DOWN))

    # ----------------------------------------------------------------------
    # Section 3 – Visual Proof Sketch (Convolution) ------------------------
    # ----------------------------------------------------------------------
    def convolution_section(self):
        title = Text("Visual Proof Sketch – Convolution as Blending", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=UP))

        # Initial simple distributions (two triangles)
        tri1 = self._make_triangle(-3, fill_color=BLUE_A)
        tri2 = self._make_triangle(1, fill_color=GREEN_A)
        for t in (tri1, tri2):
            t.scale(0.8)
        tri1.shift(LEFT*3)
        tri2.shift(RIGHT*3)
        self.play(FadeIn(tri1), FadeIn(tri2))
        self.wait(0.5)

        # First convolution → smoother shape
        conv1 = self.convolution_step(lambda x: max(0, 1-abs(x+1)),   # tri1 shifted
                                      lambda x: max(0, 1-abs(x-1)),   # tri2 shifted
                                      x_range=(-6,6), points=400)
        conv1.move_to(ORIGIN)
        self.play(TransformFromCopy(VGroup(tri1, tri2), conv1), run_time=2)
        self.wait(1)

        # Second convolution (conv1 with itself) → close to Gaussian
        conv2 = self.convolution_step(lambda x: np.exp(-x**2/2),
                                      lambda x: np.exp(-x**2/2),
                                      x_range=(-6,6), points=400)
        conv2.move_to(ORIGIN)
        self.play(Transform(conv1, conv2), run_time=2)
        self.wait(1)

        # Show the final Gaussian silhouette for comparison
        gauss = self.normal_curve(mu=0, sigma=1, length=3)
        gauss.move_to(ORIGIN)
        gauss.set_stroke(YELLOW, width=4)
        self.play(Transform(conv2, gauss), run_time=1.5)
        self.wait(1)

        self.play(FadeOut(VGroup(title, gauss), shift=DOWN))

    def _make_triangle(self, shift_x, fill_color=BLUE):
        """Utility to produce a simple triangular pdf shape."""
        points = [
            np.array([-1 + shift_x, 0, 0]),
            np.array([shift_x, 1, 0]),
            np.array([1 + shift_x, 0, 0]),
        ]
        tri = Polygon(*points, fill_opacity=0.7, fill_color=fill_color, stroke_width=1)
        return tri

    # ----------------------------------------------------------------------
    # Section 4 – Why the Normal Appears Everywhere ------------------------
    # ----------------------------------------------------------------------
    def ubiquity_section(self):
        title = Text("Why the Normal Distribution Appears Everywhere", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=UP))

        # Montage of three real‑world histograms (using random data)
        # 1. Human heights (approx normal)
        heights = np.random.normal(170, 10, 500)
        hist_h = self.histogram(heights, bins=20, bar_fill_color=GREY_A)
        hist_h.to_edge(LEFT, buff=0.5).shift(UP*0.5)

        # 2. Stock returns (heavy‑tailed but still roughly bell‑shaped)
        returns = np.random.standard_t(df=5, size=500) * 2
        hist_r = self.histogram(returns, bins=20, bar_fill_color=GREY_A)
        hist_r.next_to(hist_h, DOWN, buff=0.5)

        # 3. Test scores (bounded uniform -> becomes normal after averaging)
        scores = np.random.uniform(0, 100, 500)
        hist_s = self.histogram(scores, bins=20, bar_fill_color=GREY_A)
        hist_s.next_to(hist_r, DOWN, buff=0.5)

        self.play(FadeIn(hist_h), FadeIn(hist_r), FadeIn(hist_s))
        self.wait(2)

        # Overlay normal curves on each
        for h, data in zip([hist_h, hist_r, hist_s],
                           [heights, returns, scores]):
            mu, sigma = np.mean(data), np.std(data)
            curve = self.normal_curve(mu=mu, sigma=sigma, length=h.height)
            curve.move_to(h)
            self.play(Create(curve), run_time=1.2)

        self.wait(1)

        # Show averaging windows on a noisy signal
        signal = np.sin(np.linspace(0, 6*np.pi, 300)) + 0.5*np.random.randn(300)
        x_vals = np.linspace(0, 6*np.pi, 300)
        graph = self._plot_function(lambda x: np.sin(x) + 0.5*np.random.randn(),
                                    x_vals, color=GRAY)

        # Moving average window
        win = Rectangle(width=2, height=4, stroke_color=YELLOW, fill_opacity=0.1)
        win.to_edge(DOWN, buff=0.5)
        avg_line = VMobject(stroke_color=RED, stroke_width=3)

        def update_avg(mob, dt):
            # slide window along the signal and recompute histogram
            mob.shift(RIGHT*dt*0.5)
            # keep inside bounds
            if mob.get_right()[0] > 6*np.pi:
                mob.move_to(win.get_left())
            # compute local average histogram (quick analog)
            # Not exact: just update a tiny overlapped curve
            # Here we simply redraw a Gaussian that tightens as window moves
            # (placeholder visualisation)
        avg_line.add_updater(update_avg)

        self.play(FadeIn(win), FadeIn(avg_line))
        self.wait(3)
        avg_line.remove_updater(update_avg)

        self.play(FadeOut(VGroup(title, hist_h, hist_r, hist_s, graph, win, avg_line), shift=DOWN))

    def _plot_function(self, func, x_vals, **kwargs):
        """Return a VMobject graph of *func* over *x_vals*."""
        points = [np.array([x, func(x), 0]) for x in x_vals]
        graph = VMobject()
        graph.set_points_smoothly(points)
        graph.set_stroke(**kwargs)
        graph.scale(0.8).shift(DOWN*1.5)
        return graph

    # ----------------------------------------------------------------------
    # Section 5 – Limitations & Extensions ---------------------------------
    # ----------------------------------------------------------------------
    def limitations_section(self):
        title = Text("Limitations & Extensions", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=UP))

        # Side‑by‑side histograms: well‑behaved vs heavy‑tailed
        good = np.random.normal(0, 1, 800)
        bad = np.random.standard_cauchy(800)   # infinite variance

        hist_good = self.histogram(good, bins=30, bar_fill_color=BLUE_B)
        hist_bad = self.histogram(bad, bins=30, bar_fill_color=RED_B)

        hist_good.next_to(title, DOWN, buff=0.5, aligned_edge=LEFT)
        hist_bad.next_to(hist_good, RIGHT, buff=1.0)

        label_good = Tex(r"Finite variance", color=BLUE_B).next_to(hist_good, UP)
        label_bad = Tex(r"Infinite variance", color=RED_B).next_to(hist_bad, UP)

        self.play(FadeIn(hist_good), FadeIn(hist_bad),
                  Write(label_good), Write(label_bad))
        self.wait(2)

        # Simple flow‑chart summarising conditions
        boxes = VGroup(
            Square(side_length=1.2, fill_opacity=0.6, fill_color=GREEN).add(
                Tex(r"i.i.d.", font_size=24).move_to(ORIGIN)
            ),
            Square(side_length=1.2, fill_opacity=0.6, fill_color=GREEN).add(
                Tex(r"Finite variance", font_size=24).move_to(ORIGIN)
            ),
            Square(side_length=1.2, fill_opacity=0.6, fill_color=GREEN).add(
                Tex(r"CLT applies", font_size=24).move_to(ORIGIN)
            ),
            Square(side_length=1.2, fill_opacity=0.6, fill_color=RED).add(
                Tex(r"Heavy‑tailed\nor dependent", font_size=20).move_to(ORIGIN)
            ),
            Square(side_length=1.2, fill_opacity=0.6, fill_color=RED).add(
                Tex(r"Stable laws\n(Lévy)", font_size=20).move_to(ORIGIN)
            )
        )
        for i, b in enumerate(boxes):
            b.move_to(3*DOWN + i*RIGHT*2)
        arrows = VGroup(
            Arrow(start=boxes[0].get_right(), end=boxes[1].get_left(), buff=0),
            Arrow(start=boxes[1].get_right(), end=boxes[2].get_left(), buff=0),
            Arrow(start=boxes[0].get_bottom(), end=boxes[3].get_top(), buff=0),
            Arrow(start=boxes[3].get_right(), end=boxes[4].get_left(), buff=0),
        )
        self.play(FadeIn(VGroup(*boxes, *arrows)))
        self.wait(2)

        self.play(FadeOut(VGroup(title, hist_good, hist_bad,
                                label_good, label_bad,
                                boxes, arrows), shift=DOWN))

    # ----------------------------------------------------------------------
    # Section 6 – Takeaway -------------------------------------------------
    # ----------------------------------------------------------------------
    def takeaway_section(self):
        title = Text("Takeaway – The Power of Averaging", font_size=36).to_edge(UP)
        self.play(FadeIn(title, shift=UP))

        # Many random points appearing then coalescing into a Gaussian cloud
        cloud = VGroup()
        for _ in range(300):
            dot = Dot(radius=0.04, color=WHITE,
                      opacity=0.4).move_to(
                np.array([random.uniform(-5,5),
                          random.uniform(-5,5),
                          0]))
            cloud.add(dot)

        self.play(FadeIn(cloud, shift=UP), run_time=2)
        self.wait(1)

        # Animate moving each dot toward the centre (averaging)
        def move_to_center(mob, dt):
            direction = (ORIGIN - mob.get_center()) * dt * 0.5
            mob.shift(direction)

        for dot in cloud:
            dot.add_updater(move_to_center)

        self.wait(3)

        for dot in cloud:
           .remove_updater(move_to_center)

        # Fade in the theorem formula as the cloud stabilises
        formula = MathTex(
            r"\frac{1}{\sqrt{n}}\sum_{i=1}^{n}(X_i-\mu)\ \xrightarrow{d}\ \mathcal{N}(0,\sigma^2)",
            font_size=36,
            tex_to_color_map={"\\mathcal{N}": YELLOW}
        ).to_edge(DOWN, buff=0.7)

        self.play(Write(formula), run_time=2)
        self.wait(2)

        self.play(FadeOut(VGroup(title, cloud, formula), shift=DOWN))

    # ----------------------------------------------------------------------
    # Master construct ------------------------------------------------------
    # ----------------------------------------------------------------------
    def construct(self):
        # Run each section sequentially – feel free to comment out for testing
        self.dice_section()
        self.formal_statement_section()
        self.convolution_section()
        self.ubiquity_section()
        self.limitations_section()
        self.takeaway_section()