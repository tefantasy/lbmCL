__constant float2 e[9] = { // 9 lattice velocities
    (float2)( 0, 0),
    (float2)( 1, 0),
    (float2)( 0, 1),
    (float2)(-1, 0),
    (float2)( 0,-1),
    (float2)( 1, 1),
    (float2)(-1, 1),
    (float2)(-1,-1),
    (float2)( 1,-1)
};

__constant float w[9] = { // 9 lattice constants
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0
};

__kernel void lbm(__read_only image2d_t boundary_tex, 
                  __read_only image2d_t src_state_tex1,
                  __read_only image2d_t src_state_tex2,
                  __read_only image2d_t src_state_tex3,
                  __write_only image2d_t dst_state_tex1,
                  __write_only image2d_t dst_state_tex2,
                  __write_only image2d_t dst_state_tex3,
                  float tau,
                  int image_size_x, int image_size_y,
                  float mouse_loc_x, float mouse_loc_y)
{
    int idx_x = get_global_id(0);
    int idx_y = get_global_id(1);

    if (idx_x < image_size_x && idx_y < image_size_y) {
        const sampler_t sample = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

        float f_star[9], f_new[9];
        float2 e_norm[9];
        float2 image_size = (float2)((float)image_size_x, (float)image_size_y);
        float2 mouse_loc_norm = (float2)(mouse_loc_x + 0.5f, mouse_loc_y + 0.5f) / image_size;
        
        for (int i = 0; i < 9; i++) {
            e_norm[i] = e[i] / image_size;
        }

        int2 pos = (int2)(idx_x, idx_y);
        float2 pos_norm = (float2)((float)(idx_x + 0.5f) / (float)image_size_x, (float)(idx_y + 0.5f) / (float)image_size_y);

        f_star[0] = read_imagef(src_state_tex3, sample, pos_norm - e_norm[0]).x;
        f_star[1] = read_imagef(src_state_tex1, sample, pos_norm - e_norm[1]).x;
        f_star[2] = read_imagef(src_state_tex1, sample, pos_norm - e_norm[2]).y;
        f_star[3] = read_imagef(src_state_tex1, sample, pos_norm - e_norm[3]).z;
        f_star[4] = read_imagef(src_state_tex1, sample, pos_norm - e_norm[4]).w;
        f_star[5] = read_imagef(src_state_tex2, sample, pos_norm - e_norm[5]).x;
        f_star[6] = read_imagef(src_state_tex2, sample, pos_norm - e_norm[6]).y;
        f_star[7] = read_imagef(src_state_tex2, sample, pos_norm - e_norm[7]).z;
        f_star[8] = read_imagef(src_state_tex2, sample, pos_norm - e_norm[8]).w;

        float rho = 0.0f;
        if (distance(pos_norm, mouse_loc_norm) < 1e-3) {
            rho += 5.0f;
        }

        float2 u = (float2)(0, 0);
        for (int i = 0; i < 9; i++) {
            rho += f_star[i];
            u += f_star[i] * e[i];
        }
        u /= rho;

        float uu_dot = dot(u, u);
        for (int i = 0; i < 9; i++) {
            float eu_dot = dot(e[i], u);
            f_new[i] = w[i] * rho * (1.0f + 3.0f * eu_dot + 4.5f * eu_dot * eu_dot - 1.5f * uu_dot); // f_eq
            f_new[i] = f_star[i] - (f_star[i] - f_new[i]) / tau;
        }

        if (read_imagef(boundary_tex, sample, pos_norm).x > 0.5) {
            // Node is 'Fluid'
            write_imagef(dst_state_tex1, pos, (float4)(f_new[1], f_new[2], f_new[3], f_new[4]));
            write_imagef(dst_state_tex2, pos, (float4)(f_new[5], f_new[6], f_new[7], f_new[8]));
            write_imagef(dst_state_tex3, pos, (float4)(f_new[0], rho, u.x, u.y));
        } else {
            // Node is 'Solid'
            write_imagef(dst_state_tex1, pos, (float4)(f_star[3], f_star[4], f_star[1], f_star[2]));
            write_imagef(dst_state_tex2, pos, (float4)(f_star[7], f_star[8], f_star[5], f_star[6]));
            write_imagef(dst_state_tex3, pos, (float4)(f_star[0], rho, u.x, u.y));
        }
    }
}

__kernel void resetFluid(__write_only image2d_t state_tex1,
                         __write_only image2d_t state_tex2,
                         __write_only image2d_t state_tex3,
                         float init_rho,
                         int image_size_x, int image_size_y)
{
    // set velocity as zero, rho as init_rho, f as f_eq
    
    int idx_x = get_global_id(0);
    int idx_y = get_global_id(1);

    if (idx_x < image_size_x && idx_y < image_size_y) {
        int2 pos = (int2)(idx_x, idx_y);
        float f[9];

        for (int i = 0; i < 9; i++) {
            f[i] = w[i] * init_rho;
        }

        write_imagef(state_tex1, pos, (float4)(f[1], f[2], f[3], f[4]));
        write_imagef(state_tex2, pos, (float4)(f[5], f[6], f[7], f[8]));
        write_imagef(state_tex3, pos, (float4)(f[0], init_rho, 0, 0));
    }
}
