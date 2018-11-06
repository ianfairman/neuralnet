package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CostFunctionTest {

    @Test
    public void testL2Cost() {
        Vec wanted = new Vec(1, 2, 3);
        Vec actual = new Vec(4, -3, 7);
        double cost = new CostFunction.L2().getTotal(wanted, actual);
        assertEquals(3 * 3 + 5 * 5 + 4 * 4, cost, 0.01);

        Vec err = new CostFunction.L2().getDerivative(wanted, actual);
        assertEquals(new Vec(3, -5, 4).mul(2), err);
    }

    @Test
    public void testL2Cost2() {
        Vec wanted = new Vec(0.01, 0.99);
        Vec actual = new Vec(0.75136507, 0.77292846);
        double cost = new CostFunction.L2().getTotal(wanted, actual);
        assertEquals(0.298371109 * 2, cost, 0.01);
    }

}
