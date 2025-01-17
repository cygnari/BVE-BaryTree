#include <math.h>
#include <float.h>
#include <stdio.h>

#include "../../run_params/struct_run_params.h"
#include "user_kernel_pp.h"


void K_User_Kernel_PP(int number_of_targets_in_batch, int number_of_source_points_in_cluster,
        int starting_index_of_target, int starting_index_of_source,
        double *target_x, double *target_y, double *target_z,
        double *source_x, double *source_y, double *source_z, double *source_charge,
        struct RunParams *run_params, double *potential, int gpu_async_stream_id)
{

    double kernel_parameter=run_params->kernel_params[0];

#ifdef OPENACC_ENABLED
    #pragma acc kernels async(gpu_async_stream_id) present(target_x, target_y, target_z, \
                        source_x, source_y, source_z, source_charge, potential)
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
        for (int j = 0; j < number_of_source_points_in_cluster; j++) {

            // double dx = tx - source_x[starting_index_of_source + j];
            // double dy = ty - source_y[starting_index_of_source + j];
            // double dz = tz - source_z[starting_index_of_source + j];
            // double r  = sqrt(dx*dx + dy*dy + dz*dz);
            //
            // if (r > DBL_MIN) {
            //     temporary_potential += source_charge[starting_index_of_source + j] * exp(-kernel_parameter*r) / r;
            // }

            double sx = source_x[starting_index_of_source + j];
            double sy = source_y[starting_index_of_source + j];
            double sz = source_z[starting_index_of_source + j];
            double norm = pow(sx - tx, 2) + pow(sy - ty, 2) + pow(sz - tz, 2);

            temp = ty * sz - tz * sy;
            temp *= 1.0 / (0.5 * norm);
            temp *= source_charge[starting_index_of_source + j];
            temp *= -1.0 / (4 * M_PI);
            // temp *= source_weight[starting_index_of_source + j];
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
