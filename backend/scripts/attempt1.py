"""
Manim (Community Edition) script that visualises the Central Limit Theorem
according to the outline you provided.

To render the whole video run, for example:
    manim -pqh clt_scenes.py CLTVideo

The script defines one big scene (`CLTVideo`) that walks through the
different sections, but each section is also available as a stand-alone
scene for quick testing / reuse.
"""

from __future__ import annotations

import random
import numpy as np

from manim import (
    Scene,
    Tex,
    MathTex,
    Text,
    VGroup,
    FadeIn,
    FadeOut,
    Write,
    Create,
    Unwrite,
    Transform,
    ReplacementTransform,
    GrowFromCenter,
    Indicate,
    AddTextLetterByLetter,
    FadeTransform,
    AnimationGroup,
    TransformMatchingTex,
    BarChart,
    DashedVMobject,
    ValueTracker,
    always_redraw,
    config,
    Square,
    Circle,
    Dot,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    DL,
    DR,
    ORIGIN,
    PI,
    linear,
    Axes,
    WHITE,
    BLACK,
    BLUE,
    GREEN,
    RED,
    YELLOW,
    PURPLE,
    GRAY
)
from manim.utils.color import Colors, interpolate_color
from manim.mobject.geometry import Arrow, Line
from scipy.special import erf

# ----------------------------------------------------------------------
# Helper utilities ------------------------------------------------------
# ----------------------------------------------------------------------


def make_title(text: str) -> Tex:
    """Return a nicely formatted title."""
    return Tex(text, font_size=48).to_edge(UP)


def make_subtitle(text: str) -> Tex:
    """Return a subtitle (smaller than a title)."""
    return Tex(text, font_size=32).next_to(make_title(""), DOWN, buff=0.3)


def random_die_face() -> int:
    """Return a random integer between 1 and 6 (inclusive)."""
    return random.randint(1, 6)


def random_coin_flip() -> str:
    """Return "H" or "T"."""
    return random.choice(["H", "T"])


def dice_mobject(value: int) -> VGroup:
    """
    Very simple dice representation:
    a white square with the number in the centre.
    """
    square = Square(side_length=1, fill_color=WHITE, fill_opacity=1, stroke_color=BLACK)
    num = Tex(str(value), font_size=36).move_to(square.get_center())
    return VGroup(square, num)


def coin_mobject(face: str) -> VGroup:
    """
    Simple coin: a yellow circle with H/T inside.
    """
    circle = Circle(radius=0.5, fill_color=YELLOW, fill_opacity=1, stroke_color=BLACK)
    txt = Tex(face, font_size=36).move_to(circle.get_center())
    return VGroup(circle, txt)


def make_histogram(data: np.ndarray, bins: int, colors: list[str] | None = None) -> BarChart:
    """
    Create a BarChart that automatically normalises the data to a probability mass.
    """
    values, edges = np.histogram(data, bins=bins, density=True)
    # BarChart expects a list of values for each bar
    bars = BarChart(
        values.tolist(),
        bar_names=[f"{edges[i]:.2f}" for i in range(len(values))],
        bar_fill_opacity=0.8,
        y_range=[0, max(values) * 1.2, max(values) / 5],
        x_length=7,
        y_length=4,
        x_axis_config={"include_numbers": False},
        y_axis_config={"include_numbers": False},
        background_line_style={"stroke_opacity": 0},
    )
    if colors:
        for bar, col in zip(bars.bars, colors):
            bar.set_fill(col, opacity=0.8)
    return bars


def normal_pdf(x: np.ndarray, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """Return the values of a Normal pdf."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# ----------------------------------------------------------------------
# Individual scenes -----------------------------------------------------
# ----------------------------------------------------------------------


class IntuitionScene(Scene):
    """Introduce independent r.v.s via dice, coin flips and simple histograms."""

    def construct(self):
        title = make_title("Intuition Behind Random Variables")
        self.play(FadeIn(title, shift=UP))

        # ----- Dice animation -------------------------------------------------
        dice_label = Tex(r"Dice roll $X\sim\text{Uniform}\{1,\dots,6\}$").next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(dice_label, shift=RIGHT))

        dice = dice_mobject(random_die_face()).to_edge(LEFT, buff=1)
        self.play(GrowFromCenter(dice))
        for _ in range(6):
            new_val = random_die_face()
            new_die = dice_mobject(new_val).move_to(dice)
            self.play(Transform(dice, new_die), run_time=0.4)
        self.play(FadeOut(dice_label), FadeOut(dice))

        # ----- Coin animation -------------------------------------------------
        coin_label = Tex(r"Coin flip $Y\sim\text{Bernoulli}(p=0.5)$").next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(coin_label, shift=RIGHT))

        coin = coin_mobject(random_coin_flip()).to_edge(LEFT, buff=1)
        self.play(GrowFromCenter(coin))
        for _ in range(8):
            new_face = random_coin_flip()
            new_coin = coin_mobject(new_face).move_to(coin)
            self.play(Transform(coin, new_coin), run_time=0.3)
        self.play(FadeOut(coin_label), FadeOut(coin))

        # ----- Histograms for three base distributions -----------------------
        base_descr = Tex(
            "Base distributions:",
            r"\quad Uniform $U[0,1]$,",
            r"\quad Exponential $\mathrm{Exp}(\lambda=1)$,",
            r"\quad Bernoulli $\mathrm{Ber}(p=0.5)$",
        ).next_to(title, DOWN, buff=0.7)
        self.play(FadeIn(base_descr, shift=UP))

        # generate data
        n_samples = 500
        uniform_data = np.random.rand(n_samples)
        exponential_data = np.random.exponential(scale=1.0, size=n_samples)
        bernoulli_data = np.random.binomial(1, 0.5, size=n_samples)

        # histograms (color-coded)
        uniform_hist = make_histogram(uniform_data, bins=12, colors=[BLUE])
        exponential_hist = make_histogram(exponential_data, bins=12, colors=[GREEN])
        bernoulli_hist = make_histogram(bernoulli_data, bins=2, colors=[RED])

        # position them side-by-side
        uniform_hist.next_to(base_descr, DOWN, buff=0.8).shift(LEFT * 2)
        exponential_hist.next_to(base_descr, DOWN, buff=0.8)
        bernoulli_hist.next_to(base_descr, DOWN, buff=0.8).shift(RIGHT * 2)

        self.play(
            FadeIn(uniform_hist, shift=UP),
            FadeIn(exponential_hist, shift=UP),
            FadeIn(bernoulli_hist, shift=UP),
        )
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(base_descr),
            FadeOut(uniform_hist),
            FadeOut(exponential_hist),
            FadeOut(bernoulli_hist),
        )
        self.wait()


class SummationScene(Scene):
    """Visualise adding many independent draws – the smoothing effect."""

    def construct(self):
        title = make_title("Adding Random Variables – Visual Summation")
        self.play(FadeIn(title, shift=UP))

        # Axes for the sum distribution
        axes = Axes(
            x_range=[-5, 15, 1],
            y_range=[0, 0.25, 0.05],
            x_length=10,
            y_length=5,
            tips=False,
        )
        axes_labels = axes.get_axis_labels(x_label="Sum", y_label="Probability")
        self.play(Create(axes), Write(axes_labels))

        # Tracker for number of summands n
        n_tracker = ValueTracker(1)

        # Function returning the histogram of the sum of n iid Uniform[0,1] draws
        def get_sum_histogram():
            n = int(n_tracker.get_value())
            # generate many sample sums
            samples = np.sum(np.random.rand(20000, n), axis=1)
            hist = make_histogram(samples, bins=30, colors=[PURPLE])
            hist.move_to(axes.c2p(0, 0), aligned_edge=DL)
            # rescale to axes' coordinates
            hist.scale_to_fit_width(axes.get_width())
            return hist

        sum_hist = always_redraw(get_sum_histogram)

        # Display for n
        n_text = always_redraw(
            lambda: MathTex(f"n={int(n_tracker.get_value())}").to_corner(DR).scale(0.8)
        )
        self.play(FadeIn(sum_hist), FadeIn(n_text))

        # Animate n increasing
        self.play(
            n_tracker.animate.set_value(10),
            rate_func=linear,
            run_time=4,
        )
        self.play(
            n_tracker.animate.set_value(30),
            rate_func=linear,
            run_time=6,
        )
        self.wait(2)

        # Stacked-bar visualisation (illustrates adding layer-by-layer)
        stacked_group = VGroup()
        for i in range(1, 7):
            # one sample of i uniform draws
            sample = np.sum(np.random.rand(5000, i), axis=1)
            hist = make_histogram(sample, bins=20, colors=[interpolate_color(RED, BLUE, i / 6)])
            hist.next_to(axes, DOWN, buff=0.2 + i * 0.2)
            label = Tex(f"{i} draws").next_to(hist, UP, buff=0.2).scale(0.5)
            stacked_group.add(VGroup(hist, label))

        # Show them one after another
        for vg in stacked_group:
            self.play(FadeIn(vg), run_time=1)
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(axes_labels),
            FadeOut(sum_hist),
            FadeOut(n_text),
            *[FadeOut(vg) for vg in stacked_group],
        )
        self.wait()


class ConvergenceScene(Scene):
    """Morph histograms of different base distributions into the Gaussian."""

    def construct(self):
        title = make_title("Convergence to the Gaussian Curve")
        self.play(FadeIn(title, shift=UP))

        # Axes (shared)
        axes = Axes(
            x_range=[-4, 8, 1],
            y_range=[0, 0.5, 0.1],
            x_length=9,
            y_length=5,
            tips=False,
        )
        axes_labels = axes.get_axis_labels(x_label="Value", y_label="Density")
        self.play(Create(axes), Write(axes_labels))

        # Prepare three base distributions
        n_samples = 20000
        base_data = {
            "Uniform": np.random.rand(n_samples) * 6 - 3,          # shift to centre
            "Exponential": np.random.exponential(scale=1.0, size=n_samples) - 1,
            "Bernoulli": np.random.binomial(1, 0.5, size=n_samples) * 4 - 2,
        }

        colors = {"Uniform": BLUE, "Exponential": GREEN, "Bernoulli": RED}
        histograms = {}
        for name, data in base_data.items():
            hist = make_histogram(data, bins=30, colors=[colors[name]])
            hist.move_to(axes.c2p(0, 0), aligned_edge=DL)
            hist.scale_to_fit_width(axes.get_width())
            histograms[name] = hist

        # Normal curve (target)
        x_vals = np.linspace(-4, 8, 400)
        y_vals = normal_pdf(x_vals, mu=0, sigma=1)
        normal_curve = axes.plot_line_graph(
            x_vals, y_vals, add_vertex_dots=False, line_color=YELLOW
        )

        # Show each base histogram then morph to normal
        for name, hist in histograms.items():
            label = Tex(name).next_to(axes, UP).set_color(colors[name])
            self.play(FadeIn(hist), FadeIn(label))
            self.wait(1)
            target_label = Tex("Gaussian").next_to(axes, UP).set_color(YELLOW)
            self.play(
                ReplacementTransform(hist, normal_curve, path_arc=PI / 2),
                Transform(label, target_label),
                run_time=2,
            )
            self.wait(1)
            self.play(FadeOut(normal_curve), FadeOut(label))

        # CDF convergence overlay
        cdf_axes = Axes(
            x_range=[-4, 8, 1],
            y_range=[0, 1, 0.2],
            x_length=9,
            y_length=5,
            tips=False,
        ).next_to(axes, DOWN, buff=1)
        cdf_axes_labels = cdf_axes.get_axis_labels(x_label="Value", y_label="CDF")
        self.play(Create(cdf_axes), Write(cdf_axes_labels))

        # Empirical CDFs (for the three bases)
        def empirical_cdf(data):
            sorted_data = np.sort(data)
            cdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            return cdf_axes.plot_line_graph(sorted_data, cdf_y, add_vertex_dots=False)

        cdf_curves = {}
        for name, data in base_data.items():
            cdf_curves[name] = empirical_cdf(data).set_color(colors[name])

        # Theoretical Gaussian CDF
        def normal_cdf(x):
            return 0.5 * (1 + erf((x - 0) / (np.sqrt(2) * 1)))

        gauss_cdf_vals = normal_cdf(x_vals)
        gauss_cdf_curve = cdf_axes.plot_line_graph(
            x_vals, gauss_cdf_vals, add_vertex_dots=False, line_color=YELLOW
        )

        # Fade in all CDFs together, then highlight convergence
        self.play(
            FadeIn(VGroup(*cdf_curves.values())),
            FadeIn(gauss_cdf_curve),
            run_time=2,
        )
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(axes_labels),
            FadeOut(normal_curve),
            FadeOut(cdf_axes),
            FadeOut(cdf_axes_labels),
            FadeOut(VGroup(*cdf_curves.values())),
            FadeOut(gauss_cdf_curve),
        )
        self.wait()


class FormalStatementScene(Scene):
    """Present the theorem with animated symbols."""

    def construct(self):
        title = make_title("Formal Statement of the Central Limit Theorem")
        self.play(FadeIn(title, shift=UP))

        # Equation building blocks
        eq_parts = [
            MathTex(r"\frac{1}{\sqrt{n}}"),
            MathTex(r"\sum_{i=1}^{n}"),
            MathTex(r"\bigl( X_i - \mu \bigr)"),
            MathTex(r"\xrightarrow[n\to\infty]{d}"),
            MathTex(r"N\!\bigl(0, \sigma^{2}\bigr)."),
        ]
        eq = VGroup(*eq_parts).arrange(RIGHT, buff=0.2).scale(0.9).to_edge(UP, buff=1)

        # Animate each part appearing
        self.play(Write(eq[0]))
        self.wait(0.3)
        self.play(Write(eq[1]))
        self.wait(0.3)
        self.play(Write(eq[2]))
        self.wait(0.3)
        self.play(Write(eq[3]))
        self.wait(0.3)
        self.play(Write(eq[4]))
        self.wait(1)

        # Highlight the conditions underneath
        conditions = VGroup(
            Tex(r"$\bullet$ $(X_i)_{i\ge1}$ are i.i.d."),
            Tex(r"$\bullet$ $\mathbb{E}[X_i]=\mu$"),
            Tex(r"$\bullet$ $\operatorname{Var}(X_i)=\sigma^{2}<\infty$"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eq, DOWN, buff=0.8)

        self.play(FadeIn(conditions, shift=UP))
        self.wait(2)

        # Quick "random walk" analogy
        walk_axes = Axes(
            x_range=[0, 30, 5],
            y_range=[-10, 10, 2],
            x_length=8,
            y_length=3,
            tips=False,
        ).next_to(conditions, DOWN, buff=1)

        steps = np.cumsum(np.random.choice([-1, 1], size=30))
        walk_curve = walk_axes.plot_line_graph(
            np.arange(0, 30),
            steps,
            add_vertex_dots=False,
            line_color=WHITE,
        )
        self.play(Create(walk_axes), FadeIn(walk_curve))
        self.wait(2)

        # Fade everything out
        self.play(
            FadeOut(eq),
            FadeOut(conditions),
            FadeOut(walk_axes),
            FadeOut(walk_curve),
            FadeOut(title),
        )
        self.wait()


class UbiquityScene(Scene):
    """Explain why the normal appears in so many real-world data sets."""

    def construct(self):
        title = make_title("Why the Normal Distribution Appears Everywhere")
        self.play(FadeIn(title, shift=UP))

        # Load a real-world dataset – employ built-in random but pretend it is "heights"
        np.random.seed(0)
        heights = np.random.normal(loc=170, scale=10, size=5000)  # cm

        # Histogram of raw data
        raw_hist = make_histogram(heights, bins=30, colors=[GRAY])
        raw_hist.to_edge(LEFT, buff=1)

        raw_label = Tex(r"Raw data: Human heights").next_to(raw_hist, UP)
        self.play(FadeIn(raw_hist), Write(raw_label))

        # Fit Gaussian (using known mu, sigma)
        mu, sigma = np.mean(heights), np.std(heights)
        x_vals = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
        y_vals = normal_pdf(x_vals, mu, sigma)

        # Axes for the fitted curve
        axes = Axes(
            x_range=[mu - 30, mu + 30, 10],
            y_range=[0, max(y_vals) * 1.2, max(y_vals) / 5],
            x_length=6,
            y_length=3,
            tips=False,
        ).next_to(raw_hist, RIGHT, buff=1)
        axes_labels = axes.get_axis_labels(x_label="Height (cm)", y_label="Density")
        self.play(Create(axes), Write(axes_labels))

        # Scatter the Gaussian curve onto the same x-scale as histogram
        curve = axes.plot_line_graph(x_vals, y_vals, add_vertex_dots=False, line_color=RED)

        fit_label = Tex(r"Gaussian fit $N(\mu,\sigma^2)$").next_to(curve, UP)
        self.play(FadeIn(curve), Write(fit_label))
        self.wait(2)

        # Split-screen comparison
        split = VGroup(
            raw_hist.copy(),
            raw_label.copy(),
            curve.copy(),
            fit_label.copy(),
        )
        split.arrange_in_grid(cols=2, buff=1).shift(DOWN * 0.5)

        self.play(
            Transform(raw_hist, split[0]),
            Transform(raw_label, split[1]),
            Transform(curve, split[2]),
            Transform(fit_label, split[3]),
            run_time=1.5,
        )
        self.wait(2)

        # Clean up
        self.play(
            FadeOut(VGroup(title, raw_hist, raw_label, curve, fit_label, axes, axes_labels)),
        )
        self.wait()


class ApplicationsScene(Scene):
    """Show sliders for confidence intervals and Monte-Carlo example."""

    def construct(self):
        title = make_title("Applications and Implications")
        self.play(FadeIn(title, shift=UP))

        # ---- Confidence interval width vs. sample size -----------------------
        ci_axes = Axes(
            x_range=[10, 500, 50],
            y_range=[0, 1, 0.2],
            x_length=8,
            y_length=4,
            tips=False,
        ).to_edge(LEFT, buff=0.5)
        ci_axes_labels = ci_axes.get_axis_labels(x_label="Sample size $n$", y_label="CI half-width")
        self.play(Create(ci_axes), Write(ci_axes_labels))

        # Standard error = sigma / sqrt(n)   (assume sigma=1)
        sigma = 1.0
        n_tracker = ValueTracker(30)

        def get_ci_width(n):
            return sigma / np.sqrt(n) * 1.96  # 95% CI

        ci_curve = ci_axes.plot_line_graph(
            np.arange(10, 501, 5),
            np.array([get_ci_width(nn) for nn in np.arange(10, 501, 5)]),
            add_vertex_dots=False,
            line_color=YELLOW,
        )
        self.play(FadeIn(ci_curve))

        # Moving dot representing current n
        dot = always_redraw(
            lambda: Dot().move_to(ci_axes.c2p(n_tracker.get_value(), get_ci_width(n_tracker.get_value())))
        )
        self.play(FadeIn(dot))

        # Slider label
        n_label = always_redraw(
            lambda: MathTex(f"n={int(n_tracker.get_value())}").next_to(dot, UP)
        )
        self.play(FadeIn(n_label))

        # Animate the slider (sample size increase)
        self.play(
            n_tracker.animate.set_value(400),
            rate_func=linear,
            run_time=6,
        )
        self.wait(2)

        # ---- Monte-Carlo simulation -----------------------------------------
        mc_title = Tex("Monte-Carlo simulation of a complex system").to_edge(UP, buff=1.5)
        self.play(Write(mc_title))

        # Simulate a "complex" distribution: sum of 12 uniform(0,1) – approx normal
        sim_axes = Axes(
            x_range=[-2, 12, 2],
            y_range=[0, 0.25, 0.05],
            x_length=8,
            y_length=4,
            tips=False,
        ).next_to(ci_axes, RIGHT, buff=1)
        sim_axes_labels = sim_axes.get_axis_labels(x_label="Sum", y_label="Prob.")
        self.play(Create(sim_axes), Write(sim_axes_labels))

        # Histogram placeholder that will update as we increase sample count
        sample_tracker = ValueTracker(100)

        def monte_carlo_hist():
            N = int(sample_tracker.get_value())
            data = np.sum(np.random.rand(N, 12), axis=1)  # 12 independent draws
            hist = make_histogram(data, bins=30, colors=[PURPLE])
            hist.move_to(sim_axes.c2p(0, 0), aligned_edge=DL)
            hist.scale_to_fit_width(sim_axes.get_width())
            return hist

        mc_hist = always_redraw(monte_carlo_hist)
        self.play(FadeIn(mc_hist))

        # Animate increasing Monte-Carlo sample count
        self.play(
            sample_tracker.animate.set_value(5000),
            rate_func=linear,
            run_time=6,
        )
        self.wait(2)

        # Fade everything out
        self.play(
            FadeOut(title),
            FadeOut(ci_axes),
            FadeOut(ci_axes_labels),
            FadeOut(ci_curve),
            FadeOut(dot),
            FadeOut(n_label),
            FadeOut(mc_title),
            FadeOut(sim_axes),
            FadeOut(sim_axes_labels),
            FadeOut(mc_hist),
        )
        self.wait()


# ----------------------------------------------------------------------
# A master scene that strings everything together ----------------------
# ----------------------------------------------------------------------


class CLTVideo(Scene):
    """
    Full-length video that walks through all sections in order.
    Feel free to adjust the `run_time` parameters to match your pacing.
    """

    def construct(self):
        # 1. Intuition
        IntuitionScene.construct(self)
        self.wait(1)

        # 2. Adding / Summation
        SummationScene.construct(self)
        self.wait(1)

        # 3. Convergence to Gaussian
        ConvergenceScene.construct(self)
        self.wait(1)

        # 4. Formal statement
        FormalStatementScene.construct(self)
        self.wait(1)

        # 5. Ubiquity in real data
        UbiquityScene.construct(self)
        self.wait(1)

        # 6. Applications
        ApplicationsScene.construct(self)
        self.wait(1)

        # End fade-out
        self.play(FadeOut(*self.mobjects), run_time=2)
