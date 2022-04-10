#include <iostream>
#include <assert.h>
#include <mpi.h>


double TargetFunc(double x) {
    return 4 / (1 + x * x);
}


double Integrate(
    double lower_bound,
    double upper_bound,
    double (*Function)(double),
    int n_sections) {
        
    assert(n_sections > 0);
    
    double dx = (upper_bound - lower_bound) / n_sections;
    
    double res = 0;
    double cur_lower_bound = lower_bound;
    double cur_upper_bound = lower_bound + dx;
    
    for (int i = 0; i < n_sections; ++i) {
        res += (Function(cur_lower_bound) + Function(cur_upper_bound)) / 2 * dx;
        
        cur_lower_bound = cur_upper_bound;
        cur_upper_bound = cur_lower_bound + dx;
    }
    
    return res;
}


void HandleMainProcess(int n_processes) {
    assert(n_processes > 0);
    
    int n_sections = 0;
    std::cout << "Enter N: " << std::endl;
    std::cin >> n_sections;
    assert(n_sections > 0);
    
    const double lower_bound = 0;
    const double upper_bound = 1;
    
#ifdef MEASURE_TIME
    double singleproc_time = MPI_Wtime();
#endif
    
    double singleproc_integral = Integrate(lower_bound, upper_bound, TargetFunc, n_sections);
    
#ifdef MEASURE_TIME
    singleproc_time = MPI_Wtime() - singleproc_time;
#endif
    
#ifdef MEASURE_TIME
    double multiproc_time = MPI_Wtime();
#endif

    double multiproc_integral = 0;
    int sections_per_process = n_sections / n_processes;
    double process_full_section_len =
        (upper_bound - lower_bound) / n_sections * sections_per_process;
    
    double cur_lower_bound = lower_bound;
    double cur_upper_bound = cur_lower_bound + process_full_section_len;
    for (int i = 1; i < n_processes; ++i) {
        MPI_Send(&cur_lower_bound, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        MPI_Send(&cur_upper_bound, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
        MPI_Send(&sections_per_process, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        
        cur_lower_bound = cur_upper_bound;
        cur_upper_bound = cur_lower_bound + process_full_section_len;
    }
    
    int sections_left = n_sections % n_processes;
    double result = Integrate(
        cur_lower_bound,
        upper_bound,
        TargetFunc,
        sections_per_process + sections_left);
        
#ifndef MEASURE_TIME
    std::cout << "I_0 = " << result << std::endl;
#endif
    
    double cur_res = 0;
    for (int i = 1; i < n_processes; ++i) {
        MPI_Status status;
        MPI_Recv(&cur_res, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
        assert(status.MPI_ERROR == MPI_SUCCESS);
        
#ifndef MEASURE_TIME
        std::cout << "I_" << i << " = " << cur_res << std::endl;
#endif
        
        result += cur_res;
    }
    
#ifdef MEASURE_TIME
    multiproc_time = MPI_Wtime() - multiproc_time;
#endif
    
    std::cout << "I = " << result << std::endl;
    std::cout << "I_countinuous = " << singleproc_integral << std::endl;
    
#ifdef MEASURE_TIME
    std::cout << "Continuous integration time: " <<
        singleproc_time * 1000 << "ms" << std::endl;
    std::cout << "Multiprocess integration time: " <<
        multiproc_time * 1000 << "ms" << std::endl;
#endif
}


void HandleWorkerProcess() {
    double lower_bound = 0;
    double upper_bound = 0;
    int n_sections = 0;
    
    MPI_Status status;
    
    MPI_Recv(&lower_bound, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    assert(status.MPI_ERROR == MPI_SUCCESS);
    
    MPI_Recv(&upper_bound, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
    assert(status.MPI_ERROR == MPI_SUCCESS);
    
    MPI_Recv(&n_sections, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    assert(status.MPI_ERROR == MPI_SUCCESS);
    assert(n_sections > 0);
    
    double res = Integrate(lower_bound, upper_bound, TargetFunc, n_sections);
    
    MPI_Send(&res, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
}


int main(int argc, char* argv[]) {    
    MPI_Init(&argc, &argv);
    
    int n_processes = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    
    int process_num = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &process_num);
    
    if (process_num == 0) {
        HandleMainProcess(n_processes);
    } else {
        HandleWorkerProcess();
    }
    
    MPI_Finalize();
    return 0;
}