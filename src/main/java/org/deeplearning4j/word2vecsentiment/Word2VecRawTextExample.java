package org.deeplearning4j.word2vecsentiment;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.utilities.Params;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;

/**
 * Created by agibsonccc on 10/9/14.
 * <p>
 * Neural net that processes text into wordvectors. See below url for an in-depth explanation.
 * https://deeplearning4j.org/word2vec.html
 */
public class Word2VecRawTextExample {

  private static Logger log = LoggerFactory.getLogger(Word2VecRawTextExample.class);

  public static void main(String[] args) throws Exception {

  }

  public static void run()throws Exception {
    // Gets Path to Text file
    String filePath = "E:\\chenyuan\\dataSets\\word.txt";

    log.info("Load & Vectorize Sentences....");
    // Strip white space before and after for each line
    SentenceIterator iter = new BasicLineIterator(filePath);
    // Split on white spaces in the line to get words
    TokenizerFactory t = new DefaultTokenizerFactory();

        /*
            CommonPreprocessor will apply the following regex to each token: [\d\.:,"'\(\)\[\]|/?!;]+
            So, effectively all numbers, punctuation symbols and some special symbols are stripped off.
            Additionally it forces lower case for all tokens.
         */
    t.setTokenPreProcessor(new CommonPreprocessor());

    log.info("Building model....");
    Word2Vec vec = new Word2Vec.Builder()
            .minWordFrequency(Params.minWordFrequency)
            .iterations(Params.iterations)
            .layerSize(50)
            .seed(42)
            .windowSize(Params.windowSize)
            .iterate(iter)
            .tokenizerFactory(t)
            .build();

    log.info("Fitting Word2Vec model....");
    vec.fit();

    log.info("Writing word vectors to text file....");
    WordVectorSerializer.writeWord2VecModel(vec, Params.WORD_VECTORS_PATH);
    // Prints out the closest 10 words to "day". An example on what to do with these Word Vectors.
    log.info("Closest Words:");
    Collection<String> lst = vec.wordsNearestSum("day", 10);
    log.info("10 Words closest to 'day': {}", lst);

    // TODO resolve missing UiServer
//        UiServer server = UiServer.getInstance();
//        System.out.println("Started on port " + server.getPort());
  }
}
