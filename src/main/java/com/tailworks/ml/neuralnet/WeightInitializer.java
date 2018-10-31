package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Matrix;

public interface WeightInitializer {

    void initWeights(Matrix weights, int layer);



    // -----------------------------------------------------------------
    // --- A few predefined ones ---------------------------------------
    // -----------------------------------------------------------------
    class Random implements WeightInitializer {

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

}
