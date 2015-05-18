package org.dc.nlp

import java.io.File
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter

import scala.collection.JavaConversions._

import org.deeplearning4j.models.word2vec.Word2Vec
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache
import org.deeplearning4j.text.inputsanitation.InputHomogenization
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer

/**
 * Created by daniel on 5/17/15.
 */
 object WordVectorGenerator {
  def main(args: Array[String]) {

    val infile = new File(args(0))
    val vecfile = args(1)
    // allocate cache for approx 250 word vectors of size 50 each
    //val cache = new InMemoryLookupCache(50, 250)
    val cache = new InMemoryLookupCache()
    val sentIter = new FileSentenceIterator(new MySentPreproc(), infile)
    val tokenizer = new UimaTokenizerFactory()
    // build the Word2Vec NN and train it
    val word2vec = new Word2Vec.Builder()
      .minWordFrequency(1) // its a small corpus, every word counts
      .vocabCache(cache)
      .windowSize(10)       // build 4-grams
      .layerSize(500)      // hidden layer size
      .iterations(10)      // train for 10 epochs
      .learningRate(0.025)  // learning rate 0.1
      .iterate(sentIter)   // the custom iterator
      .tokenizerFactory(tokenizer)
      .build()
    word2vec.setVocab(cache)
    word2vec.fit()

    // do some tests on it
    val similarWordsToDay = word2vec.wordsNearest("day", 10)
    Console.println("Ten most similar words to 'day': " + similarWordsToDay)
    val similarWordsToShe = word2vec.wordsNearest("she", 1)
    Console.println("Most similar word to 'she': " + similarWordsToShe)
    val similarityHeShe = word2vec.similarity("he", "she")
    Console.println("similarity(he, she)=" + similarityHeShe)

    WordVectorSerializer.writeWordVectors(word2vec, vecfile)
    // save the transformation matrix for later use
    /*val weights = new PrintWriter(new FileWriter(wtfile), true)
    cache.vocabWords()
      .map(vocabWord => vocabWord.getWord())
      .foreach(word => weights.println("%s,%s".format(
      word2vec.getWordVector(word).map(_.toString).mkString(","), word)))
    weights.flush()
    weights.close()*/
  }

  class MySentPreproc extends SentencePreProcessor {
    override def preProcess(s: String) = new InputHomogenization(s).transform()
  }

}
