package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Matrix;

public interface Initializer {

    void initWeights(Matrix weights, int layer);


    // -----------------------------------------------------------------
    // --- A few predefined ones ---------------------------------------
    // -----------------------------------------------------------------
    class Random implements Initializer {

        java.util.Random rnd = new java.util.Random(1);

        private double min;
        private double max;

        public Random(double min, double max) {
            this.min = min;
            this.max = max;
        }

        @Override
        public void initWeights(Matrix weights, int layer) {
            double delta = max - min;
            weights.map(value -> min + rnd.nextDouble() * delta);
        }
    }


    class XavierUniform implements Initializer {
        java.util.Random rnd = new java.util.Random(1);

        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(6.0 / (weights.numberOfCols() + weights.numberOfRows()));
            weights.map(value -> (rnd.nextDouble() - 0.5) * factor);
        }
    }

    class XavierNormal implements Initializer {
        java.util.Random rnd = new java.util.Random(1);

        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = Math.sqrt(2.0 / (weights.numberOfCols() + weights.numberOfRows()));
            weights.map(value -> rnd.nextGaussian() * factor);
        }
    }

    class LeCunUniform implements Initializer {
        java.util.Random rnd = new java.util.Random(1);

        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(3.0 / weights.numberOfCols());
            weights.map(value -> (rnd.nextDouble() - 0.5) * factor);
        }
    }

    class LeCunNormal implements Initializer {
        java.util.Random rnd = new java.util.Random(1);

        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 1.0 / Math.sqrt(weights.numberOfCols());
            weights.map(value -> rnd.nextGaussian() * factor);
        }
    }

    class HeUniform implements Initializer {
        java.util.Random rnd = new java.util.Random(1);

        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = 2.0 * Math.sqrt(6.0 / weights.numberOfCols());
            weights.map(value -> (rnd.nextDouble() - 0.5) * factor);
        }
    }

    class HeNormal implements Initializer {
        java.util.Random rnd = new java.util.Random(1);

        @Override
        public void initWeights(Matrix weights, int layer) {
            final double factor = Math.sqrt(2.0 / weights.numberOfCols());
            weights.map(value -> rnd.nextGaussian() * factor);
        }
    }

}
