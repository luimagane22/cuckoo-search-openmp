#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>
#include <cassert>
#include <climits>
#include <functional>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

struct CSAParams {
    int pop_size = 4000; //tamano minimo de poblacion
    int max_iter = 500; //numero maximo de iteraciones
    double pa = 0.25; //proporcion de nidos malos que se abandonan
    double alpha = 0.01; // controla que tan grandes son los movimientos de Levy
    double beta = 1.5; //parametro de disttribucion de Levy
    int dim = 2; //dimension del problema
    int num_threads = 4; //cantidad de hilos 
    
    //para generar la animacion
    bool save_anim = false;
    int anim_every = 2; //guardar cada N iteraciones
    int anim_npts  = 150; //nidos a mostrar por frame
    std::string anim_file  = "anim_positions.dat";
};

// Generador aleatorio por hilo
static thread_local std::mt19937_64 rng_local(
    std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count()
);

inline double rand_uniform(double lo = 0.0, double hi = 1.0) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng_local);
}

inline double rand_normal(double mu = 0.0, double sigma = 1.0) {
    std::normal_distribution<double> d(mu, sigma);
    return d(rng_local);
}

inline int rand_int(int lo, int hi) {  // [lo, hi)
    std::uniform_int_distribution<int> d(lo, hi-1);
    return d(rng_local);
}

// Vuelo de Lecy con el algoritmo de Mantegna 
double levy_flight(double beta) {
    double num = std::tgamma(1.0 + beta) * std::sin(M_PI * beta / 2.0);
    double den = std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0);
    double sigma = std::pow(num / den, 1.0 / beta);
    double u = rand_normal(0.0, sigma);
    double v = rand_normal(0.0, 1.0);
    return u / std::pow(std::abs(v) + 1e-300, 1.0 / beta);
}

//Estructura para representar el nido, solucion candidata
struct Nest {
    std::vector<double> pos;
    double fitness = 1e18;
};

//Funciones objetivo 
// a) Sinusoidal: mínimo f(0,0)=0.1  dominio [-2,2]^2
double sinusoidal(const std::vector<double>& x) {
    double s = std::sin(5.0*x[0]), t = std::sin(5.0*x[1]);
    return x[0]*x[0] + x[1]*x[1] + 3.0*std::sqrt(s*s + t*t) + 0.1;
}

// b) Michalewicz: mínimo approx -1.801  dominio [0,4]^2
double michalewicz(const std::vector<double>& x) {
    double f = 0.0;
    for (int i = 0; i < (int)x.size(); ++i) {
        double t = std::sin((double)(i+1) * x[i]*x[i] / M_PI);
        f -= std::sin(x[i]) * std::pow(t, 20.0);
    }
    return f;
}

// c) Rosenbrock: mínimo f(1,1)=0  dominio [0,4]^2
double rosenbrock(const std::vector<double>& x) {
    double f = 0.0;
    for (int i = 0; i < (int)x.size()-1; ++i)
        f += std::pow(1.0 - x[i], 2.0) + 100.0*std::pow(x[i+1] - x[i]*x[i], 2.0);
    return f;
}

// d) Six-hump Camelback: mínimo approx -1.0316  dominio [-3,3]x[-2,2]
double camelback(const std::vector<double>& x) {
    double xx = x[0], yy = x[1];
    return (4.0 - 2.1*xx*xx + std::pow(xx,4.0)/3.0)*xx*xx
           + xx*yy + (-4.0 + 4.0*yy*yy)*yy*yy;
}

// Problema aplicado a cenda de cobustible
static std::vector<double> fc_i; //densidad de corriente medida
static std::vector<double> fc_v; //voltaje asociado a cada corriente

bool load_fuel_cell_data(const std::string& fname) {
    std::ifstream f(fname);
    if (!f.is_open()) return false;
    fc_i.clear(); fc_v.clear();
    double a, b;
    while (f >> a >> b) { fc_i.push_back(a); fc_v.push_back(b); }
    return !fc_i.empty();
}

inline double fc_voltage_model(double i, const std::vector<double>& th) {
    double arg = 1.0 - th[4]*i;
    if (arg <= 0.0 || i <= 0.0) return 1e10;
    return th[0] - th[1]*std::log(i) - th[2]*i + th[3]*std::log(arg);
}

double fuelcell_SSE(const std::vector<double>& th) {
    double s = 0.0;
    for (size_t k = 0; k < fc_i.size(); ++k) {
        double v = fc_voltage_model(fc_i[k], th);
        if (v > 1e9) return 1e15;
        double d = fc_v[k] - v; s += d*d;
    }
    return s;
}

double fuelcell_SAE(const std::vector<double>& th) {
    double s = 0.0;
    for (size_t k = 0; k < fc_i.size(); ++k) {
        double v = fc_voltage_model(fc_i[k], th);
        if (v > 1e9) return 1e15;
        s += std::abs(fc_v[k] - v);
    }
    return s;
}

double fuelcell_MAE(const std::vector<double>& th) {
    std::vector<double> errs;
    errs.reserve(fc_i.size());
    for (size_t k = 0; k < fc_i.size(); ++k) {
        double v = fc_voltage_model(fc_i[k], th);
        if (v > 1e9) return 1e15;
        errs.push_back(std::abs(fc_v[k] - v));
    }
    std::sort(errs.begin(), errs.end());
    size_t n = errs.size();
    return (n%2==0) ? (errs[n/2-1]+errs[n/2])/2.0 : errs[n/2];
}

//Definicion de problema
struct Problem {
    std::string name;
    int dim;
    std::vector<double> lb, ub;
    std::function<double(const std::vector<double>&)> obj;
    double known_min = 0.0;
};

Problem make_problem(const std::string& fname, const std::string& fc_mode = "SSE") {
    Problem p;
    if (fname == "sinusoidal") {
        p.name="Sinusoidal"; p.dim=2; p.lb={-2,-2}; p.ub={2,2};
        p.obj=sinusoidal; p.known_min=0.1;
    } else if (fname == "michaelwicz") {
        p.name="Michalewicz"; p.dim=2; p.lb={0,0}; p.ub={4,4};
        p.obj=michalewicz; p.known_min=-1.801;
    } else if (fname == "rosenbrock") {
        p.name="Rosenbrock"; p.dim=2; p.lb={0,0}; p.ub={4,4};
        p.obj=rosenbrock; p.known_min=0.0;
    } else if (fname == "camelback") {
        p.name="Six-hump Camelback"; p.dim=2; p.lb={-3,-2}; p.ub={3,2};
        p.obj=camelback; p.known_min=-1.0316;
    } else if (fname == "fuelcell") {
        p.name="Fuel Cell ("+fc_mode+")"; p.dim=5;
        //Limites
        p.lb={0.0,  0.0,  0.0,  0.0,  0.0};
        p.ub={0.5,  0.2,  0.15, 1.14, 60.0};
        if      (fc_mode=="SAE") p.obj=fuelcell_SAE;
        else if (fc_mode=="MAE") p.obj=fuelcell_MAE;
        else                     p.obj=fuelcell_SSE;
        p.known_min=0.0;
    } else {
        throw std::invalid_argument("Función desconocida: "+fname);
    }
    return p;
}

// CUCKOO SEARCH ALGORITHM 
struct CSAResult {
    std::vector<double> best_pos;
    double best_fit;
    std::vector<double> convergence;
    double time_seconds;
    int    iterations;
};

CSAResult cuckoo_search(const Problem& prob, const CSAParams& params) {
    const int N   = params.pop_size;
    const int D   = prob.dim;
    const double pa    = params.pa;
    const double alpha = params.alpha;
    const double beta  = params.beta;

#ifdef _OPENMP
    omp_set_num_threads(params.num_threads);
#endif

    //(A)Inicialización paralela
    // Cada nido se crea de forma independiente así que esta parte entre hilos sin dependencias entre ellos.
    std::vector<Nest> nests(N);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        nests[i].pos.resize(D);
        for (int d = 0; d < D; ++d)
            nests[i].pos[d] = rand_uniform(prob.lb[d], prob.ub[d]);
        nests[i].fitness = prob.obj(nests[i].pos);
    }

    //Buscamos cuál fue el mejor nido generado al inicio.
    int best_idx = 0;
    for (int i = 1; i < N; ++i)
        if (nests[i].fitness < nests[best_idx].fitness) best_idx = i;
    Nest best_nest = nests[best_idx];

    std::vector<double> convergence;
    convergence.reserve(params.max_iter);

    //Archivo para guardar posiciones
    std::ofstream anim_out;
    if (params.save_anim && D == 2) {
        anim_out.open(params.anim_file);
        anim_out << "# iter x y fitness best_x best_y best_f\n";
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    //Bucle principal del algoritmo
    for (int iter = 0; iter < params.max_iter; ++iter) {

        //tamaño de paso se reuce con la iteraciones
        //inicio mas exploracion y al final ajustes finos
        double alpha_t = alpha * std::exp(-3.0 * (double)iter / params.max_iter);
        alpha_t = std::max(alpha_t, alpha * 0.01);

        //(B)Vuelo de Lévy en paralelo
        // Cada nido genera un candidato nuevo. Como cada candidato se puede evaluar por separado paralelizamos
        #pragma omp parallel for schedule(dynamic, 32)
        for (int i = 0; i < N; ++i) {
            std::vector<double> new_pos(D);
            for (int d = 0; d < D; ++d) {
                double step = alpha_t * levy_flight(beta)
                            * (nests[i].pos[d] - best_nest.pos[d]);
                double np = nests[i].pos[d] + step;
                // Si el movimiento se sale del dominio lo reflejamos hacia dentro
                if (np < prob.lb[d]) np = prob.lb[d] + std::abs(np - prob.lb[d]);
                if (np > prob.ub[d]) np = prob.ub[d] - std::abs(np - prob.ub[d]);
                np = std::max(prob.lb[d], std::min(prob.ub[d], np));
                new_pos[d] = np;
            }
            double new_fit = prob.obj(new_pos);

            //nuevo candidato no reemplaza directamente a su propio nido, compite contra otro nido aleatorio
            int j = rand_int(0, N);
            if (j == i) j = (j + 1) % N;

            if (new_fit < nests[j].fitness) {
                nests[j].pos     = new_pos;
                nests[j].fitness = new_fit;
            }

            //actualizar mejor global
            if (new_fit < best_nest.fitness) {
                #pragma omp critical
                {
                    if (new_fit < best_nest.fitness) {
                        best_nest.pos     = new_pos;
                        best_nest.fitness = new_fit;
                    }
                }
            }
        }

        //(C)Abandono de nidos en paralelo
        //Primero ordenamos los nidos por calidad para identificar cuales son los peores
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b){ return nests[a].fitness < nests[b].fitness; });

        int n_ab = (int)(pa * N);
        #pragma omp parallel for schedule(static)
        for (int k = N - n_ab; k < N; ++k) {
            int ii = idx[k];
            for (int d = 0; d < D; ++d)
                nests[ii].pos[d] = rand_uniform(prob.lb[d], prob.ub[d]);
            nests[ii].fitness = prob.obj(nests[ii].pos);

            if (nests[ii].fitness < best_nest.fitness) {
                #pragma omp critical
                {
                    if (nests[ii].fitness < best_nest.fitness)
                        best_nest = nests[ii];
                }
            }
        }

        //guardar posiciones para animar cada num de iteraciones
        if (params.save_anim && D == 2 && anim_out.is_open()
            && iter % params.anim_every == 0) {
            int step = std::max(1, N / params.anim_npts);
            for (int i = 0; i < N; i += step) {
                anim_out << iter
                         << " " << nests[i].pos[0]
                         << " " << nests[i].pos[1]
                         << " " << nests[i].fitness
                         << " " << best_nest.pos[0]
                         << " " << best_nest.pos[1]
                         << " " << best_nest.fitness << "\n";
            }
        }

        convergence.push_back(best_nest.fitness);

        //si el mejor valor casi no cambia durante 100 iteraciones, se considera estancar
        //reinicio parte de los peores nidos y recupera divisibilidad
        if (iter > 100) {
            double prev = convergence[iter - 100];
            double curr = best_nest.fitness;
            if (std::abs(prev - curr) / (std::abs(prev) + 1e-12) < 1e-6) {
                int n_restart = N / 2;
                std::vector<int> idx2(N);
                std::iota(idx2.begin(), idx2.end(), 0);
                std::sort(idx2.begin(), idx2.end(),
                    [&](int a, int b){ return nests[a].fitness > nests[b].fitness; });
                #pragma omp parallel for schedule(static)
                for (int k = 0; k < n_restart; ++k) {
                    int ii = idx2[k];
                    for (int d = 0; d < D; ++d)
                        nests[ii].pos[d] = rand_uniform(prob.lb[d], prob.ub[d]);
                    nests[ii].fitness = prob.obj(nests[ii].pos);
                }
            }
        }

        if ((iter + 1) % 50 == 0 || iter == 0) {
            std::cout << "  Iter " << std::setw(4) << iter+1
                      << " | Best f = " << std::scientific << std::setprecision(6)
                      << best_nest.fitness << "\n";
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();

    CSAResult res;
    res.best_pos = best_nest.pos;
    res.best_fit = best_nest.fitness;
    res.convergence = convergence;
    res.time_seconds = std::chrono::duration<double>(t_end - t_start).count();
    res.iterations = params.max_iter;
    return res;
}

//funciones para guardar resultaods
void save_convergence(const std::string& fname, const std::vector<double>& c) {
    std::ofstream f(fname);
    f << "# iter fitness\n";
    for (size_t i = 0; i < c.size(); ++i)
        f << i+1 << " " << std::scientific << std::setprecision(10) << c[i] << "\n";
}

void save_fuelcell_curve(const std::string& fname, const std::vector<double>& th) {
    std::ofstream f(fname);
    f << "# i_density  V_calc  V_exp\n";
    for (size_t k = 0; k < fc_i.size(); ++k) {
        double Vc = fc_voltage_model(fc_i[k], th);
        f << std::fixed << std::setprecision(8)
          << fc_i[k] << "  " << Vc << "  " << fc_v[k] << "\n";
    }
}

//corremos las cuatro funciones de prueba con la misma config
void run_benchmarks(int threads) {
    CSAParams params;
    params.pop_size = 4000;
    params.max_iter = 300;
    params.num_threads = threads;

    const char* funcs[] = {"sinusoidal","michaelwicz","rosenbrock","camelback"};
    double known[] = {0.1, -1.801, 0.0, -1.0316};

    std::cout << "\n" << std::string(68,'=') << "\n";
    std::cout << "  BENCHMARK (pop=" << params.pop_size << ", iter=" << params.max_iter << ", threads=" << threads << ")\n";
    std::cout << std::string(68,'=') << "\n\n";

    for (int fi = 0; fi < 4; ++fi) {
        Problem prob = make_problem(funcs[fi]);
        std::cout << ">>> " << prob.name << "\n";
        CSAResult res = cuckoo_search(prob, params);
        std::cout << "  Resultado : f = " << std::scientific << std::setprecision(6) << res.best_fit << "  (esperado ≈ " << known[fi] << ")\n";
        std::cout << "  Posición  : [";
        for (int d = 0; d < prob.dim; ++d)
            std::cout << std::fixed << std::setprecision(5) << res.best_pos[d] << (d < prob.dim-1 ? ", " : "");
        std::cout << "]\n";
        std::cout << "  Tiempo    : " << std::fixed << std::setprecision(4) << res.time_seconds << " s\n\n";
        save_convergence(std::string("convergencia_") + funcs[fi] + ".dat", res.convergence);
    }
}

//analisis de speed up
void run_speedup_analysis(const std::string& fname) {
    CSAParams params;
    params.pop_size = 4000;
    params.max_iter = 150;
    Problem prob = make_problem(fname);

    std::cout << "\n" << std::string(60,'=') << "\n";
    std::cout << "  ANÁLISIS DE SPEED-UP: " << prob.name << "\n";
    std::cout << std::string(60,'=') << "\n";
    std::printf("  %-10s %-14s %-12s %-12s\n","Hilos","Tiempo(s)","Speed-up","Eficiencia");
    std::cout << std::string(50,'-') << "\n";

    double t_serial = 0.0;
    int thread_counts[] = {1, 2, 4, 8};

    for (int th : thread_counts) {
        params.num_threads = th;
        double total = 0.0;
        int reps = 3;
        for (int r = 0; r < reps; ++r) {
            CSAResult res = cuckoo_search(prob, params);
            total += res.time_seconds;
        }
        double avg = total / reps;
        if (th == 1) t_serial = avg;
        double speedup = t_serial / avg;
        double eff     = speedup / th * 100.0;
        std::printf("  %-10d %-14.4f %-12.3f %-10.1f%%\n", th, avg, speedup, eff);
    }
    std::cout << "\n";
}

//estimacion de parametros para combustible
// se prueba SSE SAE y MAE
void run_fuelcell(int threads) {
    if (!load_fuel_cell_data("datos_358.txt")) {
        if (!load_fuel_cell_data("datos_358.dat")) {
            std::cout << "[WARN] No se encontró datos_358.txt ni datos_358.dat. Usando datos sintéticos.\n";
        } else {
            std::cout << "[INFO] Cargados " << fc_i.size() << " puntos desde datos_358.dat\n";
        }
    } else {
        std::cout << "[INFO] Cargados " << fc_i.size() << " puntos desde datos_358.txt\n";
    }

    CSAParams params;
    params.pop_size = 8000; //mas poblacion
    params.max_iter = 1000; //mas iteraciones para convergencia fina
    params.num_threads = threads;
    params.pa = 0.20;   // menos abandono conservar buenas soluciones
    params.alpha = 0.0001; // paso muy pequeño búsqueda local fina
    params.beta = 1.5;
    params.dim = 5;

    const char* modes[] = {"SSE", "SAE", "MAE"};

    std::cout << "\n" << std::string(68,'=') << "\n";
    std::cout << "  ESTIMACIÓN DE PARÁMETROS - CELDA DE COMBUSTIBLE\n";
    std::cout << "  Referencia: E0=0.432 b=0.123 Re=0.041364 C1=0.108 C2=29.4\n";
    std::cout << std::string(68,'=') << "\n\n";

    for (auto mode : modes) {
        Problem prob = make_problem("fuelcell", mode);
        std::cout << ">>> Función objetivo: " << mode << "\n";
        CSAResult res = cuckoo_search(prob, params);

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  E0*  = " << res.best_pos[0] << "\n";
        std::cout << "  b    = " << res.best_pos[1] << "\n";
        std::cout << "  Re   = " << res.best_pos[2] << "\n";
        std::cout << "  C1   = " << res.best_pos[3] << "\n";
        std::cout << "  C2   = " << res.best_pos[4] << "\n";
        std::cout << "  f_obj= " << std::scientific << res.best_fit << "\n";
        std::cout << "  t    = " << std::fixed << std::setprecision(4) << res.time_seconds << " s\n\n";

        save_convergence(std::string("conv_fuelcell_") + mode + ".dat", res.convergence);
        save_fuelcell_curve(std::string("curva_") + mode + ".dat", res.best_pos);
    }
}

int main(int argc, char* argv[]) {
    std::string func = "all";
    int threads = 4;
    if (argc >= 2) func = argv[1];
    if (argc >= 3) threads = std::atoi(argv[2]);

    std::cout <<   "|   CUCKOO SEARCH ALGORITHM      |\n";

#ifdef _OPENMP
    std::cout << "[OpenMP] Hilos solicitados: " << threads << "\n\n";
#else
    std::cout << "Corriendo en paralelo";
#endif

    if (func == "all" || func == "benchmark") run_benchmarks(threads);
    if (func == "all" || func == "speedup")   run_speedup_analysis("rosenbrock");
    if (func == "all" || func == "fuelcell")  run_fuelcell(threads);

    // Generar datos de animacion
    if (func == "anim") {
        std::vector<std::string> anim_funcs = {
            "sinusoidal", "michaelwicz", "rosenbrock", "camelback"
        };
        for (const auto& af : anim_funcs) {
            CSAParams p;
            p.pop_size = 500; //menos nidos
            p.max_iter = 120;
            p.num_threads = threads;
            p.save_anim = true;
            p.anim_every = 2;
            p.anim_npts = 150;
            p.anim_file = "anim_" + af + ".dat";
            p.alpha = 0.05; p.pa = 0.25; p.dim = 2;
            if (af == "rosenbrock") { p.alpha = 0.02; p.pa = 0.20; }
            Problem prob = make_problem(af);
            std::cout << ">>> Animación: " << prob.name << "\n";
            CSAResult res = cuckoo_search(prob, p);
            std::cout << "  f = " << std::scientific << res.best_fit << "  -> " << p.anim_file << "\n\n";
        }
    }

    if (func=="sinusoidal"||func=="michaelwicz"||func=="rosenbrock"||func=="camelback") {
        CSAParams params;
        params.pop_size=4000; params.max_iter=300; params.num_threads=threads;
        Problem prob = make_problem(func);
        std::cout << ">>> " << prob.name << "\n";
        CSAResult res = cuckoo_search(prob, params);
        std::cout << "  f = " << std::scientific << res.best_fit << "\n  pos = [";
        for (int d=0;d<prob.dim;++d)
            std::cout << res.best_pos[d] << (d<prob.dim-1?", ":"");
        std::cout << "]\n  t = " << res.time_seconds << " s\n";
        save_convergence("conv_"+func+".dat", res.convergence);
    }

    std::cout << "\nArchivos de convergencia \n";
    return 0;
}