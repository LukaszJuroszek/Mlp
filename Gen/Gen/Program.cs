using System;

namespace Gen
{
    class Program
    {
        static void Main(string[] args)
        {
            var popLenth = 6;
            var pop = new Population(popLenth,new int[] { 1,1,1,1,0,0,0,0});
            Console.WriteLine(pop);
            Console.WriteLine("po skrzyżowaniu");
            for (int i = 0;i < 1000;i++)
            {
            pop.CrossPopulation();
            }
            Console.WriteLine(pop);
        }
    }
}
