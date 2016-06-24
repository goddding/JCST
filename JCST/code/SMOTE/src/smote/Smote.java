package smote;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Smote {
	
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
		
		int numRuns = 10;
		double[] recall=new double[numRuns];
	    double[] precision=new double[numRuns];
	    double[] fmeasure=new double[numRuns];

		double tp, fp, fn, tn;
		String classifierName[] = { "KNN"}; //{ "NBM", "KNN", "NB", "SVM"};
		
		for(int run = 0; run < numRuns; run++){
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

			
			tp = fp = fn = tn = 0;
			for (int i = 0; i < folds; i++) {
				
				Instances trains = data.trainCV(folds, i,random);
				Instances tests = data.testCV(folds, i);
							 
				//smote
				SMOTE smote=new SMOTE();
				smote.setInputFormat(trains);
				Instances smoteTrains = Filter.useFilter(trains, smote);
				
				classifier.buildClassifier(smoteTrains);				
				for (int j = 0; j < tests.numInstances(); j++) {
					
					Instance instance = tests.instance(j);
										
					double classValue = instance.classValue();					
					double result = classifier.classifyInstance(instance);
									
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
				precision[run] = tn / (tn + fn);
			if (tn + fp > 0)
				recall[run] = tn / (tn + fp);
			if (precision[run] + recall[run] > 0)
				fmeasure[run] = 2 * precision[run] * recall[run] / (precision[run] + recall[run]);
			System.out.println("The "+(run+1)+"-th run");
			System.out.println("Precision: " + precision[run]);
			System.out.println("Recall: " + recall[run]);
			System.out.println("Fmeasure: " + fmeasure[run]);
			
		}
		}
		
		double totalPrecision,totalRecall,totalFmeasure;
		totalPrecision=totalRecall=totalFmeasure=0;
		
		File file=new File("E:/compsac/"+bugType+"/"+projectName+"_smote.txt");
	    FileWriter fw=new FileWriter(file);
	    for(int run = 0; run < numRuns; run++)
	    {   
	    	totalPrecision+=precision[run];
	    	totalRecall+=recall[run];
	    	totalFmeasure+=fmeasure[run];
		    fw.write(fmeasure[run]+"\r\n");	    
	    }
	    fw.close();
	    	    
	    System.out.println("avgPrecision: " + totalPrecision/numRuns);
		System.out.println("avgRecall: " + totalRecall/numRuns);
		System.out.println("avgFmeasure: " + totalFmeasure/numRuns);
	}
}
