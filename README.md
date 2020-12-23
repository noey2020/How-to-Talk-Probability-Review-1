# How-to-Talk-Probability-Review-1

December 23, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- In order to progress with machine learning
there are three types of math that
you need to get pretty comfortable with,
probability, linear algebra, and optimization.
Of these three, I'm guessing that probability
is the one that most of you are
probably the most familiar with.
But just to be sure, what I'd like to do
is a little bit of a review broken down into three parts,
and by the time we're done with this
we'll be ready to delve much more deeply
into generative modeling for classification.
So what we'll do today is to define
the notion of the probability space for an experiment.
We'll see how to formulate an event that interests us,
and we'll see how one event influences
the probability of another event.
All of this will lead ultimately
to Bayes' rule, which is the central formula
for probabilistic reasoning in
machine learning and statistics.
So let's start with probability spaces.
A probability space for a random experiment
is a summary of all the information
we need in order to answer questions about the experiment.
Let's hear a concrete example.
So suppose we rolled two dice and we wonder,
what is the probability that they're gonna add up to 10?
So it's a random experiment, and the probability
space for this experiment has two components to it,
the set of all possible outcomes,
which is called the sample space,
and the probabilities of each of these outcomes.
So what are the outcomes for this experiment?
Well we rolled two dice so maybe the
first one is a four and the second one is a two.
We'll write it like that.
Or maybe the first one is a one and
the second die turns out to be a six.
The sample space is the set of all possible outcomes,
so it's the set of all possible pairs like this.
So the sample space consists of one, one,
one, two, all the way to one, six, then two, one,
and so on and so forth all the way to two sixes.
These are all the possible outcomes,
the set of all possible outcomes.
We could also write this a little bit more consistent,
a little bit more concisely as all possible outcomes
for the first die, set product with
all possible outcomes for the second die.
And in fact an even more concise way
of writing it is just as one, two,
three, four, five, six, squared.
Okay, so that's the sample space.
What are the probabilities of these various outcomes?
So the number of outcomes, the number
of possible outcomes is six times six, 36,
and they are all equally likely.
And so each outcome has probability 1/36.
And that is the sample space for this experiment.
Okay, so now that we have our probability space
let's formulate the event of interest.
And so the event we care about
is when the two dice add up to 10,
and we're wondering what is the probability of this event?
So let's call the event A.
An event is just a subset of the possible outcomes.
So it's a subset of the possible outcomes.
And in particular, the event consists
of all pairs, z1, z2 let's say, that add up to 10.
Okay, so which pairs are these?
Well if the first die is a one
then the second one has to be a nine, so that doesn't work.
So the first die actually has to be at least a four.
Okay, so if the first one is a four
then the second one has to be a six,
if the first one is a five then the
second one has to be a five, and if the
first one is a six then the second one
has to be a four, and that's it.
So this is the event that interests us.
It consists of three possible outcomes out of the 36.
So if we were to draw a Venn diagram for this
we have the set of all 36 outcomes
and the event that interests us,
which consists of just three out of these events.
And what is the probability of it?
Well the three outcomes, and each of them
has probability 1/36, and so the
probability of A is just 1/12.
So that's how we define an event
and compute its probability.
Now when we are using probabilities
in statistics and machine learning,
we're typically dealing not just with
one event but with multiple events.
For instance, we measure a patient's temperature
and blood pressure, and wonder whether
the patient has a particular disease.
So that's a coming together of three events.
There's the event that the patient
has that particular blood pressure,
there's the event that the patient
has a particular temperature, and there's the
event that the patient has the disease.
So we get to observe two of these events,
the temperature and the blood pressure,
and we wonder what they tell us probabilistically
about the third event, the one we don't
get to observe, the one we care about, the disease.
So let's see a concrete example of this type.
So here's a situation, a toy situation.
So you have 10 coins in front of you
and they all look the same but they're not the same.
Nine of them are regular fair coins,
coins that come up heads half the time
and tails half the time, but the tenth coin
is a bad coin that always comes up tails.
So you close your eyes and you pick a coin at random,
and then you toss the coin four times
and you find that it comes up tails every single time.
What is the probability that you picked the bad coin?
So this is something that's not obvious
and we're gonna have to work this out.
So first off, what is the sample
space for this random experiment?
So what's going on?
We first pick a coin at random
and then we toss it four times.
Okay, so the way I'm going to write down
an outcome is first I'll say which coin we picked,
just the coin number, coin number one to 10.
Let's say 10 is the bad coin.
And then what happened on the first toss,
heads or tails, and the second toss,
the third toss, and the fourth toss.
That's how we'll describe a particular outcome,
and so the set of all possible outcomes
is the set of all choices for the coin,
there are 10 possible choices, product with the
set of all possibilities for toss one
and toss two, and three and four.
This is the sample space of possible outcomes.
In fact, we can write this a little bit more concisely.
So we can also write it as one,
all coin choices, times H T to the fourth.
So how many possible outcomes are there?
Well there's 10 times two times two
times two times two, so 160 possible outcomes.
Okay, so that's the sample space.
So now the event we care about
is whether or not we got the bad coin.
So let's call that A.
So A is the event that we picked the bad coin.
And formally what the event is,
is the set of all outcomes in which
the coin we picked was the bad one, coin number 10,
and then it doesn't matter what happens afterwards.
So this is the event we care about,
but it's not all we get to observe.
We get to observe a different event which is related,
and that's that all the coins come up tails.
And so here we don't know what the coin number is,
but all the outcomes are tails.
So these are the two events that we're dealing with.
What is the probability that both of them occur?
Okay, so what is the probability, A and B?
And the way we usually write this,
in fact, is A intersection B.
What is the probability that both of these things happen?
Well it's the probability that we picked the bad coin
times the probability that we get all tails
given the bad coin, given that the coin is bad.
So the probability of picking the bad coin is 1/10
since there's one bad coin and nine good coins.
And once we've picked the bad coin
the probability that all four tosses
are gonna come up tails, that is one.
So the probability that both A and B happen is 1/10.
Okay, so this has all been fairly intuitive
but we are using some general rules
about conditioning over here, and so let's be
a little bit more explicit about that.
So when we have two events, A and B,
we can talk about the conditional
probability of B given A, the probability
that B occurs given that we know A occurs.
And the notation for that involves this bar over here.
This bar means given.
Now the most basic formula for conditional
probability is this one shown over here.
It says that the probability that
A and B both occur is the probability
that A occurs times the probability
that B occurs given that A has occurred.
So returning to our example, there are
two events that we're dealing with, A and B.
A is that the bad coin is chosen,
B is that we have all tails, and we are
interested in the probability of A given B.
We've seen that B is true and we're wondering
what is the probability of A given this information.
Well we can compute it directly from this formula.
So if we rearrange terms we see that the
probability of A given B is the probability
of A and B divided by the probability of B.
So what is the probability of A and B?
We've computed that, that's 1/10.
What is the probability of B?
What is the probability that all four tosses come up tails?
Well that's something we have to figure out.
Okay, so let's go ahead and do that.
Okay, so the probability that all four tosses are tails,
let's figure that out.
Well it kind of depends on whether or not
we chose the bad coin, so let's
break it up into those two cases.
So it's the probability that we got the bad coin
and we get all tails, plus the probability
that we did not get the bad coin
and yet we still get all tails.
So now let's compute these.
The probability that we get the bad coin is 1/10.
If we get the bad coin, the probability
of all tails is one because the
bad coin always comes up tails.
The probability that we don't get the bad coin
is 9/10 because there are nine good coins.
And if we don't get the bad coin,
if we have a fair coin, the probability
of getting all tails, of getting four tails in a row,
is 1/2 times 1/2 times 1/2 times 1/2, so it's 1/16.
Okay, so let's work this out.
1/10 plus 9/160 is 25/160,
which is 5/32.
So this is the probability of B,
the probability of all tails, and now
let's go back to what we actually wanted,
the probability of A given B.
And as we saw, it's the probability
of A and B over the probability of B.
The probability of A and B we figured out, that was 1/10.
The probability of B we just figured out, it's 5/32.
Okay, and so this works out to 32/50,
which is 64/100 or 0.64.
So there is exactly a 64% probability
that we have the bad coin.
Pretty impressive, huh?
So when we were doing those calculations
we were implicitly using Bayes' rule,
which is the central formula for probabilistic reasoning
in statistics and machine learning.
So let's spell out this formula
a little bit more precisely now.
We have two events, A and B, and the event
we are interested in is event A,
but we don't get to observe it directly.
We get to observe some other event B,
and what we're interested in is given that B happened,
what does it tell us about the probability of A happening?
So here is Bayes' rule.
If we don't observer anything about B,
the probability that A happens is just probability of A,
but now that we know that B happened,
that changes the probability of A.
By how much?
By a multiplicative correction factor,
and this is the correction factor.
Now this formula is just two applications
of the general rule for conditioning.
So let's see how we did it.
The probability of A given B is as we know,
just the probability of A and B over the probability of B.
And now let's just work on the numerator.
The probability of A and B is just
the probability of A times the probability of B given A.
And we keep the denominator the same,
and sure enough we have Bayes' rule.
The various calculations we've done today
might seem very simple but they're actually very powerful.
Bayes' rule tells us how we can reason
under uncertainty, and it's no surprise, therefore,
that it's one of the fundamental formulas
for inference in machine learning.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
