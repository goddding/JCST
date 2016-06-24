package oversampling;

import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;

public class OverSampling {
	
	public static void overSampling(Instances instances){
		
		int pos, neg;
		pos = neg = 0;

		for (Instance instance : instances) {
			if (instance.classValue() == 0.0)
				neg++;
			else
				pos++;
		}
		
		int tmpsize = instances.numInstances();

		Random random = new Random(47);
		while (neg > pos) {

			int i = random.nextInt(tmpsize);

			Instance instance = instances.get(i);
			if (instance.classValue() == 1.0) {
				instances.add(instance);
				pos++;
			}

		}
		
	}

}

