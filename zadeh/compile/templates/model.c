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

double clip(double value){
    if (value < 0)
        return 0;
    if (value > 1)
        return 1;
    return value;
}


double gauss(double x, double s, double a){
    return exp(- pow((x-a)/s, 2.0) / 2.0);
}


double gauss2(double x, double s1, double a1, double s2, double a2){
    if (a1 <= x && x<=a2)
        return 1.0;
    if (x < a1)
        return gauss(x, s1, a1);
    return gauss(x, s2, a2);
}


double s_shaped(double x, double a, double b){
    if (x <= a)
        return 0.0;
    if (x >= b)
        return 1.0;
    if (x <= (a + b) / 2.0)  // (a, (a+b)/2]
        return 2 * pow(((x - a) / (b - a)), 2);
    // ((a+b)/2, b)
    return 1 - 2 * pow(((x - b) / (b - a)),  2);
}


double z_shaped(double x, double a, double b){
    if (x <= a)
        return 1.0;
    if (x >= b)
        return 0.0;
    if (x <= (a + b) / 2.0)  // (a, (a+b)/2]
        return 1 - 2 * pow(((x - a) / (b - a)), 2);
    // ((a+b)/2, b)
    return 2 * pow(((x - b) / (b - a)),  2);
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
