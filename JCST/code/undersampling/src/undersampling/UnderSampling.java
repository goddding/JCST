package undersampling;

import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class UnderSampling {

	public static void underSampling(Instances instances) {
		int pos, neg;
		pos = neg = 0;

		for (Instance instance : instances) {
			if (instance.classValue() == 0.0)
				neg++;
			else
				pos++;
		}

		Random random = new Random(47);
		while (neg > pos) {

			int i = random.nextInt(instances.numInstances());

			Instance instance = instances.get(i);
			if (instance.classValue() == 0.0) {
				instances.delete(i);
				neg--;
			}

		}

	}

}

