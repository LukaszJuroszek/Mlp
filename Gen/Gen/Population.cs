using System;
using System.Collections.Generic;
using System.Linq;

namespace Gen
{
    public class Population
    {
        private static Random rand;
        public List<IndividualPerson> CurrentPopulation { get; private set; }
        public int[] Template { get; private set; }
        public Population(int lenth,int[] template)
        {
            rand = new Random();
            CurrentPopulation = new List<IndividualPerson>();
            Template = template;
            CurrentPopulation = GeneratePopulation(lenth);
            CoutPopulationMarks();
        }
        private List<IndividualPerson> GeneratePopulation(int lenth)
        {
            var result = new List<IndividualPerson>();
            for (var i = 0;i < lenth;i++)
                result.Add(new IndividualPerson(Template.Length,Template));
            return result;
        }
        private void CoutPopulationMarks()
        {
            CurrentPopulation.ForEach(x => x.CountMarks(Template));
        }
        private int GetAllMarks() => CurrentPopulation.Sum(x => x.Mark);
        private int GetRandomParentIndex()
        {
            var randomNumberFromPopulation = rand.Next(0,GetAllMarks());
            var currentSum = 0;
            for (var i = 0;i < CurrentPopulation.Count;i++)
            {
                currentSum += CurrentPopulation[i].Mark;
                if (currentSum >= randomNumberFromPopulation)
                    return i;
            }
            return CurrentPopulation.Count;
        }
        public void CrossPopulation()
        {

        }
    }
}
