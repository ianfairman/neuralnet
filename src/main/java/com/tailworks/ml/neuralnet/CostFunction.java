package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;

// FIXME harmonise w Activation
public interface CostFunction {

    String getName();
    double getTotal(Vec wanted, Vec actual);
    Vec getDerivative(Vec wanted, Vec actual);

    // --------------------------------------------------------------

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
