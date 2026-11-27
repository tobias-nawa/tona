// Calibrated acceptance probabilities of the drafter's drafts, online without training (like vLLM PR 52228,
// vllm.ai/blog/2026-10-14-adaptive-verification-online-estimator): p_i = sigmoid(omega * logit(q_i) + beta_i) with
// q_i the drafter's probability for its draft at position i, one slope omega for all positions, one intercept beta_i
// per position. Fitted by IRLS (Newton) over the last N observations, with a ridge towards the start (omega = 1,
// beta = 0: the raw q). Observation: position i accepted (1) or first rejected (0), nothing behind it.
// Measured (research/measurements/2026-10-25-barriers): raw q on prose gives 1.78 expected against 1.42 accepted
// per pass.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace tona::spec {

template <int K>
class DraftCalib {
public:
    static constexpr int P = K + 1;  // omega, beta_0 ... beta_{K-1}

    explicit DraftCalib(int window = 256, int refit = 16, double ridge = 1.0) : n_(window), m_(refit), ridge_(ridge) {
        th_.fill(0);
        th_[0] = 1;
        prior_ = th_;
    }
    // Start from pre-fitted parameters (omega, beta_0, ...); the ridge pulls towards them instead of the raw q
    explicit DraftCalib(const std::array<double, P>& prior, int window = 256, int refit = 16, double ridge = 1.0)
        : n_(window), m_(refit), ridge_(ridge), th_(prior), prior_(prior) {}

    double predict(double q, int i) const { return sigm(th_[0] * logit(q) + th_[1 + i]); }

    // q: probabilities of the K drafts, k: how many were checked, a: how many were accepted (a <= k)
    void observe(const double* q, int k, int a) {
        for (int i = 0; i < k && i <= a; i++) {
            Obs o{logit(q[i]), i, i < a};
            if ((int)obs_.size() < n_)
                obs_.push_back(o);
            else
                obs_[head_] = o, head_ = (head_ + 1) % n_;
        }
        if (++passes_ % m_ == 0) fit();
    }

    const std::array<double, P>& params() const { return th_; }

private:
    struct Obs {
        double x;
        int pos;
        bool y;
    };
    static double sigm(double v) { return 1 / (1 + std::exp(-v)); }
    static double logit(double q) {
        q = std::clamp(q, 1e-4, 1 - 1e-4);
        return std::log(q / (1 - q));
    }

    struct Normal {
        double H[P][P] = {}, g[P] = {};
    };

    // Normal equations of the ridge-regularized logistic fit at th
    Normal normal_equations(const std::array<double, P>& th) const {
        Normal N;
        for (int a = 0; a < P; a++) N.H[a][a] = ridge_, N.g[a] = -ridge_ * (th[a] - prior_[a]);
        for (const Obs& o : obs_) {
            const int j = 1 + o.pos;
            const double p = sigm(th[0] * o.x + th[j]), w = p * (1 - p), r = (o.y ? 1.0 : 0.0) - p;
            N.H[0][0] += w * o.x * o.x, N.H[0][j] += w * o.x, N.H[j][0] += w * o.x, N.H[j][j] += w;
            N.g[0] += r * o.x, N.g[j] += r;
        }
        return N;
    }

    // Newton step H^-1 g by Gauss-Jordan with pivoting, P <= 5
    static std::array<double, P> newton_step(const Normal& N) {
        double A[P][P + 1];
        for (int a = 0; a < P; a++) {
            for (int b = 0; b < P; b++) A[a][b] = N.H[a][b];
            A[a][P] = N.g[a];
        }
        for (int c = 0; c < P; c++) {
            int piv = c;
            for (int r = c + 1; r < P; r++)
                if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
            for (int b = 0; b <= P; b++) std::swap(A[c][b], A[piv][b]);
            for (int r = 0; r < P; r++) {
                if (r == c) continue;
                const double f = A[r][c] / A[c][c];
                for (int b = c; b <= P; b++) A[r][b] -= f * A[c][b];
            }
        }
        std::array<double, P> step;
        for (int a = 0; a < P; a++) step[a] = A[a][P] / A[a][a];
        return step;
    }

    // Plain Newton ran away when almost only accepted drafts were seen (q ~ 1, omega up to 1,400): halve the step
    // until the objective does not decrease. Returns false (th unchanged) if no step size works.
    bool line_search(std::array<double, P>& th, const std::array<double, P>& step) const {
        std::array<double, P> cand;
        const double f0 = objective(th);
        double t = 1;
        bool ok = false;
        for (int h = 0; h < 20 && !ok; h++, t /= 2) {
            for (int a = 0; a < P; a++) cand[a] = th[a] + t * step[a];
            ok = objective(cand) >= f0;
        }
        if (ok) th = cand;
        return ok;
    }

    void fit() {
        std::array<double, P> th = th_;
        for (int it = 0; it < 5; it++)
            if (!line_search(th, newton_step(normal_equations(th)))) break;
        for (double v : th)
            if (!std::isfinite(v)) return;  // the old state stays
        th[0] = std::clamp(th[0], 0.05, 4.0);
        for (int a = 1; a < P; a++) th[a] = std::clamp(th[a], -8.0, 8.0);
        th_ = th;
    }

    // Log-likelihood with the ridge towards the start value
    double objective(const std::array<double, P>& th) const {
        double f = 0;
        for (int a = 0; a < P; a++) f -= 0.5 * ridge_ * (th[a] - prior_[a]) * (th[a] - prior_[a]);
        for (const Obs& o : obs_) {
            const double v = th[0] * o.x + th[1 + o.pos];
            // log sigmoid(v) resp. log(1 - sigmoid(v)), numerically stable
            f += o.y ? -std::log1p(std::exp(-v)) : -std::log1p(std::exp(v));
        }
        return f;
    }

    int n_, m_;
    double ridge_;
    std::array<double, P> th_, prior_;
    std::vector<Obs> obs_;
    int head_ = 0;
    long passes_ = 0;
};

}  // namespace tona::spec
