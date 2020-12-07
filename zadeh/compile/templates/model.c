#include <stdarg.h>
#include <stdlib.h>
#include <math.h>


double min(int arg_count, ...)
{
    int i;
    double min, a;

    va_list ap;
    va_start(ap, arg_count);
    min = va_arg(ap, double);
    for (i = 2; i <= arg_count; i++)
        if ((a = va_arg(ap, double)) < min)
            min = a;
    va_end(ap);

    return min;
}

double max(int arg_count, ...)
{
    int i;
    double max, a;

    va_list ap;
    va_start(ap, arg_count);
    max = va_arg(ap, double);
    for (i = 2; i <= arg_count; i++)
        if ((a = va_arg(ap, double)) > max)
            max = a;
    va_end(ap);

    return max;
}


double mean(double *values, int n){
    double sum = 0;
    int i;
    for(i = 0; i < n; i++)
        sum+=values[i];
    return sum/n;
}

double weighted_mean(double *values, double* weights, int n){
    double sum = 0;
    double sum_weights = 0;
    int i;
    for(i = 0; i < n; i++){
        sum+=values[i]*weights[i];
        sum_weights +=weights[i];
    }
    return sum/sum_weights;
}



double {{name}}(double {{target}}, {{inputs_typed}}){
    return {{code}};
}


double {{name}}_crisp(double min_val, double max_val, int n, {{inputs_typed}}){
    double* values = (double*) malloc(n*sizeof(double));
    double* weights = (double*) malloc(n*sizeof(double));
    double increment= (max_val-min_val) / n;
    double x=min_val;
    double result;
    int i;


    for (i=0; i<n;i++){
        x+=increment;
        values[i]=x;
        weights[i]={{name}}(x, {{inputs_listed}});
    }

    result = weighted_mean(values, weights, n);
    free(values);
    free(weights);

    return result;
}
