package idshuffle;

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import java.util.Random;

import java.io.IOException;
import java.util.Iterator;

public class IdShuffle extends Configured implements Tool{

    public static class IdMapper extends Mapper<LongWritable, Text, Text, Text>{
        final static Random random = new Random();
        
        public void map(LongWritable offset, Text line, Context context) throws IOException, InterruptedException {
            context.write(new Text(Integer.toString(random.nextInt(1000))), line);
        }
    }

    public static class IdReducer extends Reducer<Text, Text, Text, NullWritable>{
        public void reduce(Text line, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Iterator<Text> it = values.iterator();
            while (it.hasNext()){
                context.write(it.next(), NullWritable.get());
            }
        }
    }

    @Override
    public int run(String[] strings) throws Exception {
        Path outputPath = new Path(strings[1]);

        Job job1 = Job.getInstance();
        job1.setJarByClass(IdShuffle.class);

        job1.setMapperClass(IdMapper.class);
        job1.setReducerClass(IdReducer.class);

        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(NullWritable.class);

        job1.setInputFormatClass(TextInputFormat.class);
        job1.setOutputFormatClass(TextOutputFormat.class);

        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(Text.class);

        job1.setNumReduceTasks(8);

        TextInputFormat.addInputPath(job1, new Path(strings[0]));
        TextOutputFormat.setOutputPath(job1, outputPath);

        return job1.waitForCompletion(true)? 0: 1;
    }

    public static void main(String[] args) throws Exception {
        new IdShuffle().run(args);
    }
}
