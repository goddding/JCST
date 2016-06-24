package costsensitive;

import java.io.File;
import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CostSensitiveClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class CostSensitive {
	
	public static void main(String agrs[]) throws Exception{
		String bugType="Surprise";
		String projectName = "wicket";		
		String fileRootPath = "../../datasets/"+bugType+"/"+projectName+".arff";		
		Instances rawData = DataSource.read(fileRootPath);
				
		StringToWordVector filter = new StringToWordVector(10000);
		filter.setInputFormat(rawData);
		String[] options = { "-W", "10000", "-L", "-M", "2",
				"-stemmer", "weka.core.stemmers.IteratedLovinsStemmer", 
				"-stopwords-handler", "weka.core.stopwords.Rainbow", 
				"-tokenizer", "weka.core.tokenizers.AlphabeticTokenizer" };

		filter.setOptions(options);
		Instances data = Filter.useFilter(rawData, filter);
		data.setClassIndex(0);

		int NumPositive, NumNegtive;
		NumPositive = NumNegtive = 0;
		for (Instance instance : data) {
			if (instance.classValue() == 0.0)
				NumNegtive++;
			else
				NumPositive++;
		}
		
		
		double precision, recall, fmeasure;
		double tp, fp, fn, tn;
		String classifierName[] = { "KNN"}; //{ "NBM", "KNN", "NB", "SVM"};
		for(String name:classifierName){
			
			System.out.println(name);
			Classifier classifier = null;
			
			if (name.equals("NBM"))
				classifier = new NaiveBayesMultinomial();

			if (name.equals("NB"))
				classifier = new NaiveBayes();

			if (name.equals("KNN"))
				classifier = new IBk();

			if (name.equals("SVM"))
				classifier = new SMO();

			//cross validation
			int folds = 10;			
			Random random = new Random(1);
			data.randomize(random);
			data.stratify(folds);

			//cost-sensitive
			precision = recall = fmeasure = 0;
			tp = fp = fn = tn = 0;
			for (int i = 0; i < folds; i++) {
				
				Instances trains = data.trainCV(folds, i,random);
				Instances tests = data.testCV(folds, i);
					
				CostMatrix costMatrix = new CostMatrix(2);	
				costMatrix.setElement(0, 1, NumPositive);
				costMatrix.setElement(1, 0, NumNegtive);
				CostSensitiveClassifier csClassifier = new CostSensitiveClassifier();
				csClassifier.setClassifier(classifier);
				csClassifier.setCostMatrix(costMatrix);
				csClassifier.buildClassifier(trains);
				
				for (int j = 0; j < tests.numInstances(); j++) {
					
					Instance instance = tests.instance(j);
										
					double classValue = instance.classValue();					
					double result = csClassifier.classifyInstance(instance);
									
					if (result == 0.0 && classValue == 0.0) {
						tp++;
					} else if (result == 0.0 && classValue == 1.0) {
						fp++;
					} else if (result == 1.0 && classValue == 0.0) {
						fn++;
					} else if (result == 1.0 && classValue == 1.0) {
						tn++;
					}
				}	
			}
					
			if (tn + fn > 0)
				precision = tn / (tn + fn);
			if (tn + fp > 0)
				recall = tn / (tn + fp);
			if (precision + recall > 0)
				fmeasure = 2 * precision * recall / (precision + recall);
			System.out.println("costSensitive");
			System.out.println("Precision: " + precision);
			System.out.println("Recall: " + recall);
			System.out.println("Fmeasure: " + fmeasure);
			
		}
	}

}


