package org.deeplearning4j.utilities;
/**
 * @ClassName: CSVUtils
 * @Description: TODO
 * @Author: Ricardo
 * @Date: 2019/5/25 20:42
 **/

import com.csvreader.CsvWriter;
import org.deeplearning4j.word2vecsentiment.TestData;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * CSV操作(读取和写入)
 * * @author lq
 * * @version 2018-04-23   */


public class CSVUtils {

  public static void writeCsvFile(String csvFilePath, List<TestData> testDataList){

    try {
      // 创建CSV写对象 例如:CsvWriter(文件路径，分隔符，编码格式);
      CsvWriter csvWriter = new CsvWriter(csvFilePath, ',', Charset.forName("UTF-8"));
      // 写内容
      String[] headers = {"ID","Pred"};
      csvWriter.writeRecord(headers);
      for(int i=0;i<testDataList.size();i++){
        csvWriter.writeRecord(testDataList.get(i).toStrings());
      }

      csvWriter.close();
      System.out.println("--------CSV文件已经写入--------");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}