#include <math.h>
#include <float.h>
#include <stdio.h>

#include "../../run_params/struct_run_params.h"
#include "user_kernel_pc.h"


void K_User_Kernel_PC_Lagrange(int number_of_targets_in_batch, int number_of_interpolation_points_in_cluster,
        int starting_index_of_target, int starting_index_of_cluster,
        double *target_x, double *target_y, double *target_z,
        double *cluster_x, double *cluster_y, double *cluster_z, double *cluster_charge,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    double kernel_parameter = run_params->kernel_params[0];

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        cluster_x, cluster_y, cluster_z, cluster_charge, potential)
    {
#endif
#ifdef OPENACC_ENABLED
    #pragma acc loop independent
#endif
    for (int i = 0; i < number_of_targets_in_batch; i++) {

        double temporary_potential = 0.0;
        double temp;

        double tx = target_x[starting_index_of_target + i];
        double ty = target_y[starting_index_of_target + i];
        double tz = target_z[starting_index_of_target + i];

#ifdef OPENACC_ENABLED
        #pragma acc loop independent reduction(+:temporary_potential)
#endif
        for (int j = 0; j < number_of_interpolation_points_in_cluster; j++) {

            double cx = cluster_x[starting_index_of_cluster + j];
            double cy = cluster_y[starting_index_of_cluster + j];
            double cz = cluster_z[starting_index_of_cluster + j];
            double norm = pow(cx - tx, 2) + pow(cy - ty, 2) + pow(cz - tz, 2);

            temp = ty * cz - tz * cy;
            temp *= 1.0 / (0.5 * norm);
            temp *= cluster_charge[starting_index_of_cluster + j];
            temp *= -1.0 / (4 * M_PI);
            // temp *= cluster_weight[starting_index_of_cluster + j];
            // double dx = tx - cluster_x[starting_index_of_cluster + j];
            // double dy = ty - cluster_y[starting_index_of_cluster + j];
            // double dz = tz - cluster_z[starting_index_of_cluster + j];
            // double r  = sqrt(dx*dx + dy*dy + dz*dz);
            //
            // temporary_potential += cluster_charge[starting_index_of_cluster + j] * exp(-kernel_parameter * r) / r;
            temporary_potential += temp;

        } // end loop over interpolation points
#ifdef OPENACC_ENABLED
        #pragma acc atomic
#endif
        potential[starting_index_of_target + i] += temporary_potential;
    }
#ifdef OPENACC_ENABLED
    } // end kernel
#endif
    return;
}
