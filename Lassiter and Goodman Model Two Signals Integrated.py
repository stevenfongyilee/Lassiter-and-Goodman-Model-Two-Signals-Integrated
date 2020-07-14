# need to make this faster by substituting CDF for integrals.

import numpy
numpy.set_printoptions(linewidth = 120)
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import uniform
from scipy import integrate

###################

def receiver_0_signal_0(h):
	return state_distribution.pdf(h)

def receiver_0_signal_1(h, theta):
	if h < theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))
		
def sender_1_signal_0_non_normalized(h):
	return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h)) - 0))

def sender_1_signal_1_non_normalized(h, theta):
	if h < theta:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))) - cost))
		
def sender_1_signal_0_normalized(h, theta):
	if h < theta:	
		return 1.
	else:
		return sender_1_signal_0_non_normalized(h) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta))

def sender_1_signal_1_normalized(h, theta):
	if h < theta:
		return 0.
	else:
		return sender_1_signal_1_non_normalized(h, theta) / (sender_1_signal_0_non_normalized(h) + sender_1_signal_1_non_normalized(h, theta))

# def receiver_2_signal_0_non_normalized(h, theta):
# 	if h  < theta:
# 		return theta_distribution.pdf(theta) * state_distribution.pdf(h)
# 	else:
# 		return sender_1_signal_0_normalized(h, theta) * theta_distribution.pdf(theta) * state_distribution.pdf(h)

# def receiver_2_signal_0_non_normalized(h, theta):
# 	return ((numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h)) - 0))) / ((numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h)) - 0))) + sender_1_signal_1_non_normalized(h, theta))) * theta_distribution.pdf(theta) * state_distribution.pdf(h)

def receiver_2_signal_0_non_normalized_h_gte_theta_1(h, theta_1):
	numerator = ((1. - state_distribution.cdf(theta_1)) * numpy.exp(cost)) ** choice_parameter
	return (numerator / (numerator + 1)) * theta_distribution.pdf(theta_1) * state_distribution.pdf(h)

def receiver_2_signal_0_non_normalized_h_lt_theta_1(h, theta_1):
	return theta_distribution.pdf(theta_1) * state_distribution.pdf(h)

# def receiver_2_signal_1_non_normalized(h, theta):
# 	if h < theta:
# 		return 0.
# 	else:
# 		return sender_1_signal_1_normalized(h, theta) * theta_distribution.pdf(theta) * state_distribution.pdf(h)
	
def receiver_2_signal_1_non_normalized(h, theta_1):
	denominator = ((1. - state_distribution.cdf(theta_1)) * numpy.exp(cost)) ** choice_parameter
	return ((denominator + 1) ** -1) * theta_distribution.pdf(theta_1) * state_distribution.pdf(h)

def receiver_2_signal_0_normalized_h(h, receiver_2_signal_0_normalization_factor):
	return (integrate.quad(lambda theta_1, h : receiver_2_signal_0_non_normalized_h_gte_theta_1(h, theta_1), lower_bound, h, args = (h))[0] + integrate.quad(lambda theta_1, h : receiver_2_signal_0_non_normalized_h_lt_theta_1(h, theta_1), h, upper_bound, args = (h))[0]) / receiver_2_signal_0_normalization_factor

def receiver_2_signal_1_normalized_h(h, receiver_2_signal_1_normalization_factor):
	return integrate.quad(lambda theta_1, h : receiver_2_signal_1_non_normalized(h, theta_1), lower_bound, h, args = (h))[0] / receiver_2_signal_1_normalization_factor

def receiver_2_signal_1_normalized_theta(theta_1, receiver_2_signal_1_normalization_factor):
	return integrate.quad(receiver_2_signal_1_non_normalized, theta_1, upper_bound, args = (theta_1))[0] / receiver_2_signal_1_normalization_factor

##########################################################################################

# Here we have the settings for a level 0 receiver decoding probabilities, given a fixed
# theta. This forms the common basis for both Lassiter and Goodman's original model and
# our modified model.

cost = 2.
choice_parameter = 4.
lower_bound = 0.
upper_bound = 1.
num_states = 80

# mu = 0.
# sigma = 1.
# state_distribution = norm(mu,sigma)

alpha_parameter = 1.
beta_parameter = 9.
location_parameter = lower_bound
scale_parameter = upper_bound - lower_bound
state_distribution = beta(alpha_parameter, beta_parameter, loc = location_parameter, scale = scale_parameter)

# state_distribution = uniform(lower_bound, upper_bound - lower_bound)

theta_distribution_type = 'uniform'
if theta_distribution_type == 'normal':
	theta_distribution = norm(mu, sigma)
elif theta_distribution_type == 'Beta':
	theta_distribution = beta(3, 3, loc = lower_bound, scale = upper_bound - lower_bound)
elif theta_distribution_type == 'uniform':
	theta_distribution = uniform(lower_bound, upper_bound - lower_bound)

array_0 = numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - ((numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - numpy.linspace(lower_bound, upper_bound, num_states, endpoint = False))/2)

#########################


# theta_1_distribution_array = theta_distribution.pdf(array_0)/((upper_bound - lower_bound)/len(array_0))
theta_1_distribution_array = theta_distribution.pdf(array_0)
theta_1_distribution_array = theta_1_distribution_array / numpy.sum(theta_1_distribution_array)

fix, ax = pyplot.subplots(1,1)
pyplot.plot(theta_1_distribution_array)
pyplot.show()

#########################

sender_1_signal_0_non_normalized_array = numpy.empty(0)
for h_num in range(len(array_0)):
	value = sender_1_signal_0_non_normalized(array_0[h_num])
	sender_1_signal_0_non_normalized_array = numpy.append(sender_1_signal_0_non_normalized_array, value)

# print 'sender_1_signal_0_non_normalized_array = \n%s' % sender_1_signal_0_non_normalized_array

sender_1_signal_1_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_signal_1_non_normalized(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_signal_1_non_normalized_array = numpy.insert(sender_1_signal_1_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_signal_1_non_normalized_array = \n%s' % sender_1_signal_1_non_normalized_array

#########################

denominator_array = numpy.tile(sender_1_signal_0_non_normalized_array, (len(array_0), 1)) + sender_1_signal_1_non_normalized_array

sender_1_signal_0_normalized_array = numpy.tile(sender_1_signal_0_non_normalized_array, (len(array_0), 1)) / denominator_array
sender_1_signal_1_normalized_array = sender_1_signal_1_non_normalized_array / denominator_array

sender_1_signal_0_normalized_array = sender_1_signal_0_normalized_array * numpy.reshape(theta_1_distribution_array, (len(array_0), 1))
sender_1_signal_1_normalized_array = sender_1_signal_1_normalized_array * numpy.reshape(theta_1_distribution_array, (len(array_0), 1))

print sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array
print numpy.sum(sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array)

#########################

sender_1_signal_0_h_array = numpy.sum(sender_1_signal_0_normalized_array, axis = 0)
sender_1_signal_1_h_array = numpy.sum(sender_1_signal_1_normalized_array, axis = 0)

#########################

fixed_theta_1_num = numpy.int(numpy.ceil(len(array_0)*(2./3.)))

print 'fixed_theta_1 = %s' % array_0[fixed_theta_1_num]

sender_1_signal_1_fixed_theta_1_h_array = sender_1_signal_1_normalized_array[fixed_theta_1_num]
sender_1_signal_1_fixed_theta_1_h_array = sender_1_signal_1_fixed_theta_1_h_array / theta_1_distribution_array[fixed_theta_1_num]

print 'sender_1_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_signal_1_fixed_theta_1_h_array

#########################

receiver_2_signal_0_normalization_factor = integrate.dblquad(lambda theta_1, h : receiver_2_signal_0_non_normalized_h_gte_theta_1(h, theta_1), lower_bound, upper_bound, lambda x: lower_bound, lambda x: x)[0] + integrate.dblquad(lambda theta_1, h : receiver_2_signal_0_non_normalized_h_lt_theta_1(h, theta_1), lower_bound, upper_bound, lambda x: x, lambda x: upper_bound)[0]
# receiver_2_signal_0_normalization_factor = 0.987667752268
print 'receiver_2_signal_0_normalization_factor = %s' % receiver_2_signal_0_normalization_factor

receiver_2_signal_1_normalization_factor = integrate.dblquad(lambda theta_1, h : receiver_2_signal_1_non_normalized(h, theta_1), lower_bound, upper_bound, lambda x: lower_bound, lambda x: x)[0]
# receiver_2_signal_1_normalization_factor = 0.0123322182507
print 'receiver_2_signal_1_normalization_factor = %s' % receiver_2_signal_1_normalization_factor

receiver_2_signal_0_h_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_0_h_array = numpy.append(receiver_2_signal_0_h_array, receiver_2_signal_0_normalized_h(h, receiver_2_signal_0_normalization_factor))

receiver_2_signal_1_h_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_1_h_array = numpy.append(receiver_2_signal_1_h_array, receiver_2_signal_1_normalized_h(h, receiver_2_signal_1_normalization_factor))

receiver_2_signal_1_theta_1_array = numpy.empty(0)
for h in array_0:
	receiver_2_signal_1_theta_1_array = numpy.append(receiver_2_signal_1_theta_1_array, receiver_2_signal_1_normalized_theta(h, receiver_2_signal_1_normalization_factor))

#########################

fig, ax = pyplot.subplots(1, 2, figsize = (12,5))

pyplot.subplot(1, 2, 1)
line = pyplot.plot(array_0, sender_1_signal_0_h_array, lw = 2, color = 'k')
line = pyplot.plot(array_0, sender_1_signal_1_h_array, lw = 2, color = 'b')

line = pyplot.plot(array_0, sender_1_signal_1_fixed_theta_1_h_array, lw = 2, linestyle = '--', color = 'b')

line = pyplot.plot(array_0, sender_1_signal_1_normalized_array[:,fixed_theta_1_num] / theta_1_distribution_array, lw = 5, linestyle = ':', color = 'b')

pyplot.subplot(1, 2, 2)
line = pyplot.plot(array_0, receiver_2_signal_0_h_array, lw = 2, color = 'k')
line = pyplot.plot(array_0, receiver_2_signal_1_h_array, lw = 2, color = 'b')
line = pyplot.plot(array_0, receiver_2_signal_1_theta_1_array, lw = 2, linestyle = '--', color = 'b')

line = pyplot.plot(array_0, receiver_0_signal_0(array_0), color = 'r')

pyplot.subplot(1, 2, 1)
pyplot.legend([r'$\sigma_{1}(u_{0}|h)$', r'$\sigma_{1}(u_{1}|h)$', r'$\sigma_{1}(u_{1}|h, \theta_{1} \approx %s)$' % numpy.around(array_0[fixed_theta_1_num], decimals = 2)], loc = 0, fontsize = 14)

pyplot.subplot(1, 2, 2)
pyplot.legend([r'$\rho_{2}(h|u_{0})$', r'$\rho_{2}(h|u_{1})$', r'$\rho_{2}(\theta_{1}|u_{1})$'], loc = 0, fontsize = 14)

fig.text(.4, 0, r'Lassiter and Goodman Two Signals Integrated\n', fontsize = 10)

# fig.text(.4, 0, r'$\lambda = %s, C(u_{1}) = %s, \mu = %s, \sigma = %s, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, mu, sigma, num_states, theta_distribution_type), fontsize = 14)
fig.text(.4, 0, r'$\lambda = %s, C(u_{1}) = %s, \alpha = %s, \beta = %s, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, alpha_parameter, beta_parameter, num_states, theta_distribution_type), fontsize = 14)
# fig.text(.4, 0, r'$\lambda = %s, C(u_{1}) = %s, Uniform distribution, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, num_states, theta_distribution_type), fontsize = 14)

# pyplot.savefig('Lassiter and Goodman Model Two Signals Normal Distribution.pdf')
# pyplot.savefig('Lassiter and Goodman Model Two Signals Beta Distribution.pdf')
# pyplot.savefig('Lassiter and Goodman Model Two Signals Uniform Distribution.pdf')

pyplot.show()
pyplot.close()