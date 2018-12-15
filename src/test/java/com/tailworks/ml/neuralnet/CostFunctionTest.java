package com.tailworks.ml.neuralnet;

import com.tailworks.ml.neuralnet.math.Vec;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class CostFunctionTest {

    @Test
    public void testQuadraticCost() {
        Vec expected = new Vec(1, 2, 3);
        Vec actual = new Vec(4, -3, 7);
        double cost = new CostFunction.Quadratic().getTotal(expected, actual);
        assertEquals(3 * 3 + 5 * 5 + 4 * 4, cost, 0.01);

        Vec err = new CostFunction.Quadratic().getDerivative(expected, actual);
        assertEquals(new Vec(3, -5, 4).mul(2), err);
    }

    @Test
    public void testQuadraticCost3() {
        Vec expected = new Vec(1, 0.2);
        Vec actual = new Vec(0.712257432295742, 0.533097573871501);
        double cost = new CostFunction.Quadratic().getTotal(expected, actual);
        assertEquals(0.19374977898811957, cost, 0.0000001);
    }

}
