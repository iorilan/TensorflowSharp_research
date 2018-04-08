using System;
using System.Diagnostics;
using System.IO;
using TensorFlow;

namespace Ts_ImageClassification
{
    /// <summary>
    /// 1. install Tensorflowsharp
    /// 2.download tensorflowlib.dll from 
    /// http://ci.tensorflow.org/view/Nightly/job/nightly-libtensorflow-windows/lastSuccessfulBuild/artifact/lib_package/libtensorflow-cpu-windows-x86_64.zip
    /// </summary>
    class Program
    {
        
        static void Main(string[] args)
        {
           
            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();

            var graph = new TFGraph();
            var model = File.ReadAllBytes("../../ml_model/model.pb");
            var labels = File.ReadAllLines("../../ml_model/labels.txt");
            graph.Import(model);

            var bestIdx = 0;
            float best = 0;


            var paths = Directory.GetFiles("../../img");

            using (var session = new TFSession(graph))
            {
                foreach (var path in paths)
                {
                    var tensor = ImageUtil.CreateTensorFromImageFile(path);
                    var runner = session.GetRunner();
                    runner.AddInput(graph["Placeholder"][0], tensor).Fetch(graph["loss"][0]);
                    var output = runner.Run();
                    var result = output[0];

                    var probabilities = ((float[][])result.GetValue(jagged: true))[0];
                    for (int i = 0; i < probabilities.Length; i++)
                    {
                        if (probabilities[i] > best)
                        {
                            bestIdx = i;
                            best = probabilities[i];
                        }
                    }




                    // fin
                    stopwatch.Stop();
                    Console.WriteLine($"result of image [{path}] = {labels[bestIdx]} ({best * 100.0}%)");
                    Console.WriteLine($"Total time: {stopwatch.Elapsed}");
                    Console.ReadKey();

                }


            }

        }
    }
}
