using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

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
                result.Add(new IndividualPerson(Template,Template.Length));
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
        private int GetIndividualSize() => CurrentPopulation.First().GetIndividualSize() + 1;
        private void CrossTwoPerson()
        {
            var firstPersonIndex = GetRandomParentIndex();
            var secondPersonIndex = GetRandomParentIndex();
            var splitingNumber = rand.Next(0,GetIndividualSize());
            CrossTwoPersonBySplitingNumber(firstPersonIndex,secondPersonIndex,splitingNumber);
        }
        private void CrossTwoPersonBySplitingNumber(int firstPersonIndex,int secondPersonIndex,int splitingNumber)
        {
            var firstPersonAfterCrossing = CrossPersonWithSecondBySplitingNumber(splitingNumber,CurrentPopulation[firstPersonIndex],CurrentPopulation[secondPersonIndex]);
            var secondPersonAfterCrossing = CrossPersonWithSecondBySplitingNumber(splitingNumber,CurrentPopulation[secondPersonIndex],CurrentPopulation[firstPersonIndex]);
            CurrentPopulation[firstPersonIndex].SetIndividual(firstPersonAfterCrossing,Template);
            CurrentPopulation[secondPersonIndex].SetIndividual(secondPersonAfterCrossing,Template);
        }
        private List<int> CrossPersonWithSecondBySplitingNumber(int splitingNumber,IndividualPerson firstPerson,IndividualPerson secondPerson)
        {
            var result = new List<int>();
            foreach (var item in firstPerson.GetIndividual().Take(splitingNumber))
                result.Add(item);
            foreach (var item in secondPerson.GetIndividual().Skip(splitingNumber))
                result.Add(item);
            return result;
        }
        public void CrossPopulation()
        {
            foreach (var item in Enumerable.Range(1,CurrentPopulation.Count))
            {
                CrossTwoPerson();
            }
            CoutPopulationMarks();
        }
        public override string ToString()
        {
            var st = new StringBuilder();
            foreach (var item in CurrentPopulation)
            {
                st.AppendLine(item.ToString());
            }
            st.Append($"Mark Sum:{GetAllMarks()}, Template ");
            foreach (var item in Template)
            {
                st.Append(item+" ");
            }
            return st.ToString();
        }
    }
}
