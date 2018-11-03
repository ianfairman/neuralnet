package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Function;
import com.tailworks.ml.neuralnet.math.Vec;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.util.Arrays.stream;

public class Activation {

    private String name;
    private Function fn;
    private Function dFn;

    public Activation(String name) {
        this.name = name;
    }

    public Activation(String name, Function fn, Function dFn) {
        this.name = name;
        this.fn = fn;
        this.dFn = dFn;
    }

    public Vec fn(Vec vec) {
        return vec.map(fn);
    }

    public Vec dFn(Vec vec) {
        return vec.map(dFn);
    }

    public String getName() {
        return name;
    }






    // -----------------------------------------------------------------
    // --- A few predefined ones ---------------------------------------
    // -----------------------------------------------------------------

    public static Activation ReLU = new Activation(
            "ReLU",
            x -> x <= 0 ? 0 : x,                // σ
            x -> x <= 0 ? 0 : 1                 // σ'
    );

    public static Activation Leaky_ReLU = new Activation(
            "Leaky_ReLU",
            x -> x <= 0 ? 0.01 * x : x,         // σ
            x -> x <= 0 ? 0.01 : 1              // σ'
    );

    public static Activation LogSigmoid = new Activation(
            "LogSigmoid",
            x -> 1.0 / (1.0 + exp(-x)),         // σ
            x -> x * (1.0 - x)                  // σ'
    );

    public static Activation Softplus = new Activation(
            "Softplus",
            x -> log(1.0 + exp(x)),             // σ
            x -> 1.0 / (1.0 + exp(-x))          // σ'
    );

    public static Activation Identity = new Activation(
            "Identity",
            x -> x,             // σ
            x -> 1              // σ'
    );


    // Softmax needs a little extra love since element output depends on more than
    // same element input. Simple element mapping will not suffice.
    public static Activation Softmax = new Activation("Softmax") {
        @Override
        public Vec fn(Vec vec) {
            double[] data = vec.getData();
            double sum = 0;
            for (double a : data)
                sum += exp(a);

            double finalSum = sum;
            return vec.map(value -> exp(value) / finalSum);
        }

        @Override
        public Vec dFn(Vec vec) {
            throw new NotImplementedException();
        }
    };

    public static Activation Softmax_broken = new Activation("Softmax_broken") {
        @Override
        public Vec fn(Vec vec) {
            double sum = stream(vec.getData()).sum();
            return vec.map(value -> value / sum);
        }

        @Override
        public Vec dFn(Vec vec) {
            throw new NotImplementedException();
        }
    };
}
