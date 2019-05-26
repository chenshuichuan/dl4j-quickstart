package org.deeplearning4j.word2vecsentiment;

import com.csvreader.CsvReader;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

/**
 * This is a DataSetIterator that is specialized for the IMDB review dataset used in the Word2VecSentimentRNN example
 * It takes either the train or test set data from this data set, plus a WordVectors object (typically the Google News
 * 300 pretrained vectors from https://code.google.com/p/word2vec/) and generates training data sets.<br>
 * Inputs/features: variable-length time series, where each word (with unknown words removed) is represented by
 * its Word2Vec vector representation.<br>
 * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
 *
 * @author Alex Black
 */
public class SentimentIterator implements DataSetIterator {
  private final WordVectors wordVectors;
  private final int batchSize;
  private final int vectorSize;
  private final int truncateLength;

  private int cursor = 0;
  private  String[] positiveFiles;
  private  String[] negativeFiles;
  private final TokenizerFactory tokenizerFactory;

  /**
   * @param dataDirectory  the directory of the IMDB review data set
   * @param wordVectors    WordVectors object
   * @param batchSize      Size of each minibatch for training
   * @param truncateLength If reviews exceed
   * @param train          If true: return the training data. If false: return the testing data.
   */
  public SentimentIterator(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException {
    this.batchSize = batchSize;
    this.vectorSize = wordVectors.getWordVector(wordVectors.vocab().wordAtIndex(0)).length;
    File file = new File(dataDirectory);
    Scanner sc = new Scanner(file);
    List<String> posList = new ArrayList<>();
    List<String> negList = new ArrayList<>();

    while (sc.hasNextLine()){
      String dataLine = sc.nextLine();
      if(dataLine!=null){
        String[] datas = dataLine.split(",");
        if(3==datas.length){
          if(datas[1].isEmpty()){
            System.out.println(datas[0]+":"+datas[1]+":"+datas[2]);
          }
          if("Positive".equals(datas[2])){
            posList.add(datas[1]);
          }
          else if("Negative".equals(datas[2])){
            negList.add(datas[1]);
          }
        }
      }
    }

    positiveFiles =  new String[posList.size()];
    posList.toArray(positiveFiles);
    negativeFiles =  new String[negList.size()];
    negList.toArray(negativeFiles);

    this.wordVectors = wordVectors;
    this.truncateLength = truncateLength;

    tokenizerFactory = new DefaultTokenizerFactory();
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
  }


  @Override
  public DataSet next(int num) {
    if (cursor >= positiveFiles.length + negativeFiles.length) {
      throw new NoSuchElementException();
    }
    try {
      return nextDataSet(num);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private DataSet nextDataSet(int num) throws IOException {
    //First: load reviews to String. Alternate positive and negative reviews
    List<String> reviews = new ArrayList<>(num);
    boolean[] positive = new boolean[num];
    for (int i = 0; i < num && cursor < totalExamples(); i++) {
      if (cursor % 2 == 0) {
        //Load positive review
        int posReviewNumber = cursor / 2;
        if(posReviewNumber<positiveFiles.length){
          String review = positiveFiles[posReviewNumber];
          reviews.add(review);
          positive[i] = true;
        }
        else{
          //Load negative review
          int negReviewNumber = cursor / 2;
          String review = negativeFiles[negReviewNumber];
          reviews.add(review);
          positive[i] = false;
        }
      } else {
        //Load negative review
        int negReviewNumber = cursor / 2;
        if(negReviewNumber<negativeFiles.length){
          String review = negativeFiles[negReviewNumber];
          reviews.add(review);
          positive[i] = false;
        }
        else{
          //Load positive review
          int posReviewNumber = cursor / 2;
          String review = positiveFiles[posReviewNumber];
          reviews.add(review);
          positive[i] = true;
        }
      }
      cursor++;
    }

    //Second: tokenize reviews and filter out unknown words
    List<List<String>> allTokens = new ArrayList<>(reviews.size());
    int maxLength = 0;
    for (String s : reviews) {
      List<String> tokens = tokenizerFactory.create(s).getTokens();
      List<String> tokensFiltered = new ArrayList<>();
      for (String t : tokens) {
        if (wordVectors.hasWord(t)){
          tokensFiltered.add(t);
        }
      }
      if(tokensFiltered.size()==0){
        System.out.println("s=0:"+s);
      }
      allTokens.add(tokensFiltered);
      maxLength = Math.max(maxLength, tokensFiltered.size());
    }

    //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
    if (maxLength > truncateLength){
      maxLength = truncateLength;
    }

    //Create data for training
    //Here: we have reviews.size() examples of varying lengths
    INDArray features = Nd4j.create(new int[]{reviews.size(), vectorSize, maxLength}, 'f');
    INDArray labels = Nd4j.create(new int[]{reviews.size(), 2, maxLength}, 'f');    //Two labels: positive or negative
    //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
    //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
    INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
    INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

    for (int i = 0; i < reviews.size(); i++) {
      List<String> tokens = allTokens.get(i);
      if(tokens.size()==0){
        System.out.println(this.cursor+":"+i);
        //tokens.add("null");
      }
      // Get the truncated sequence length of document (i)
      int seqLength = Math.min(tokens.size(), maxLength);

      // Get all wordvectors for the current document and transpose them to fit the 2nd and 3rd feature shape
      final INDArray vectors = wordVectors.getWordVectors(tokens.subList(0, seqLength)).transpose();

      // Put wordvectors into features array at the following indices:
      // 1) Document (i)
      // 2) All vector elements which is equal to NDArrayIndex.interval(0, vectorSize)
      // 3) All elements between 0 and the length of the current sequence
      features.put(
              new INDArrayIndex[]{
                      NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.interval(0, seqLength)
              },
              vectors);

      // Assign "1" to each position where a feature is present, that is, in the interval of [0, seqLength)
      featuresMask.get(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.interval(0, seqLength)}).assign(1);

      int idx = (positive[i] ? 0 : 1);
      int lastIdx = Math.min(tokens.size(), maxLength);
      labels.putScalar(new int[]{i, idx, lastIdx - 1}, 1.0);   //Set label: [0,1] for negative, [1,0] for positive
      labelsMask.putScalar(new int[]{i, lastIdx - 1}, 1.0);   //Specify that an output exists at the final time step for this example
    }

    return new DataSet(features, labels, featuresMask, labelsMask);
  }


  public int totalExamples() {
    return positiveFiles.length + negativeFiles.length;
  }

  @Override
  public int inputColumns() {
    return vectorSize;
  }

  @Override
  public int totalOutcomes() {
    return 2;
  }

  @Override
  public void reset() {
    cursor = 0;
  }
  @Override
  public boolean resetSupported() {
    return true;
  }

  @Override
  public boolean asyncSupported() {
    return true;
  }

  @Override
  public int batch() {
    return batchSize;
  }


  public int cursor() {
    return cursor;
  }


  public int numExamples() {
    return totalExamples();
  }

  @Override
  public void setPreProcessor(DataSetPreProcessor preProcessor) {
    throw new UnsupportedOperationException();
  }

  @Override
  public List<String> getLabels() {
    return Arrays.asList("positive", "negative");
  }

  @Override
  public boolean hasNext() {
    return cursor < numExamples();
  }

  @Override
  public DataSet next() {
    return next(batchSize);
  }

  @Override
  public void remove() {

  }

  @Override
  public DataSetPreProcessor getPreProcessor() {
    throw new UnsupportedOperationException("Not implemented");
  }

  /**
   * Convenience method for loading review to String
   */
  public String loadReviewToString(int index) throws IOException {
    String str;
    if (index % 2 == 0){
      str = positiveFiles[index / 2];
    }
    else{
      str = negativeFiles[index / 2];
    }
    return str;
  }

  /**
   * Convenience method to get label for review
   */
  public boolean isPositiveReview(int index) {
    return index % 2 == 0;
  }


  public INDArray loadFeaturesFromFile(String review, int maxLength) throws IOException {
    return loadFeaturesFromString(review, maxLength);
  }

  public INDArray loadFeaturesFromString(String reviewContents, int maxLength) {
    List<String> tokens = tokenizerFactory.create(reviewContents).getTokens();
    List<String> tokensFiltered = new ArrayList<>();
    for (String t : tokens) {
      if (wordVectors.hasWord(t)){
        tokensFiltered.add(t);
      }
    }
    int outputLength = Math.max(maxLength, tokensFiltered.size());
    INDArray features = Nd4j.create(1, vectorSize, outputLength);
    for (int j = 0; j < tokens.size() && j < maxLength; j++) {
      String token = tokens.get(j);
      INDArray vector = wordVectors.getWordVectorMatrix(token);
      features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
    }
    return features;
  }

  public int getCursor() {
    return cursor;
  }

  public void setCursor(int cursor) {
    this.cursor = cursor;
  }
}
