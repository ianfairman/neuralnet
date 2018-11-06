package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;

// FIXME harmonise w Activation
public interface CostFunction {

    String getName();

    double getTotal(Vec wanted, Vec actual);

    Vec getDerivative(Vec wanted, Vec actual);


    // -----------------------------------------------------------------
    // --- A few standard cost functions -------------------------------
    // -----------------------------------------------------------------

    // −∑[E * ln aLj+(1−Erj) ln (1−aLj)]
    /*
    class CrossEntropy implements CostFunction {

        @Override
        public Vec getDerivative(Vec wanted, Vec actual) {
            double[] y = wanted.getData();
            double[] o = actual.getData();
            double sum = 0;
            for (int i = 0; i < o.length; i++)
                sum += y[i] * log10(o[i]) + (1.0 - y[i]) * log10(1.0 - o[i]);           // fixme: change to ln ist.

            return new Vec();//-sum /* y.length*/;          // fixme: strictly speaking this divide should be here ... restore after backprop is implemented
//        }
    //}


    // 1/N * ∑(L−E)^2
    class MSE implements CostFunction {
        @Override
        public String getName() {
            return "MSE";
        }

        @Override
        public double getTotal(Vec wanted, Vec actual) {
            Vec diff = wanted.sub(actual);
            return diff.dot(diff) / actual.dimension();
        }

        @Override
        public Vec getDerivative(Vec wanted, Vec actual) {
            return actual.sub(wanted).mul(2.0 / actual.dimension());
        }
    }

    // ∑(L−E)^2
    class L2 implements CostFunction {
        @Override
        public String getName() {
            return "L2";
        }

        @Override
        public double getTotal(Vec wanted, Vec actual) {
            Vec diff = wanted.sub(actual);
            return diff.dot(diff);
        }

        @Override
        public Vec getDerivative(Vec wanted, Vec actual) {
            return actual.sub(wanted).mul(2);
        }
    }

    // 0.5 * ∑(L−E)^2
    class L2Half implements CostFunction {
        @Override
        public String getName() {
            return "L2Half";
        }

        @Override
        public double getTotal(Vec wanted, Vec actual) {
            Vec diff = wanted.sub(actual);
            return diff.dot(diff)*0.5;
        }

        @Override
        public Vec getDerivative(Vec wanted, Vec actual) {
            return actual.sub(wanted);
        }
    }


}
