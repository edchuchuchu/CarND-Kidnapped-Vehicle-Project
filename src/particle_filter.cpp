/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random> // Need this for sampling from distributions
#include "particle_filter.h"
#include <limits>
#include <map>

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// random engine initialized
	default_random_engine gen;
	// Set the number of particles.
	num_particles = 1000;
	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	// Set standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// Creates a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (unsigned int i = 0; i < num_particles; ++i) {
		Particle particle;
		// Initialize all particles to first position
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(particle.weight);
		//cout << "Sample " << " " << particle.id << " " << particle.x << " " << particle.y << " " << particle.theta << " " << particle.weight << " "<< endl;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// random engine initialized
	default_random_engine gen;
	// Standard deviations for x, y, and theta
	double std_x, std_y, std_theta;
	// Set standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	for (Particle &particle: particles){
		double current_theta = particle.theta;
		double new_theta = current_theta + yaw_rate * delta_t;
		if (yaw_rate != 0){
			// Add measurements to the particle
			particle.x += velocity / yaw_rate * (sin(new_theta) - sin(current_theta));
			particle.y += velocity / yaw_rate * (cos(current_theta) - cos(new_theta));
			particle.theta = new_theta;
			// Creates a random Gaussian noise
			normal_distribution<double> dist_x(particle.x, std_x);
			normal_distribution<double> dist_y(particle.y, std_y);
			normal_distribution<double> dist_theta(particle.theta, std_theta);
			// Add a random Gaussian noise
			particle.x = dist_x(gen);
			particle.y = dist_y(gen);
			particle.theta = dist_theta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (LandmarkObs &obs : observations){
    	double closet_dist = std::numeric_limits<double>::max();
    	for(LandmarkObs pred: predicted){
    		double distance = dist(pred.x, pred.y, obs.x, obs.y);
    		if (distance < closet_dist){
    			closet_dist = distance;
    			obs.id = pred.id;
    		}
    	}
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	// Set Covariance
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	for (unsigned int i = 0; i < particles.size(); ++i) {
		Particle particle = particles[i];
		// Create predicted LandmarkObs
		std::vector<LandmarkObs> predicted;
		// Create predicted LandmarkObs LookUp Table
		map<int, LandmarkObs> predMap;
		for (Map::single_landmark_s  landmark:map_landmarks.landmark_list){
			if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range){
				LandmarkObs pred;
				pred= {landmark.id_i, landmark.x_f, landmark.y_f};
				predicted.push_back(pred);
				predMap.insert({pred.id, pred});
			}
		}

		// Transformations
		std::vector<LandmarkObs> trans_observations;
		for (LandmarkObs obs:observations){
			LandmarkObs trans_obs;
			trans_obs.id = obs.id;
			trans_obs.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
			trans_obs.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
			trans_observations.push_back(trans_obs);
		}

		// Associations
		dataAssociation(predicted, trans_observations);

		// Calculating the Particle's Final Weight
		double weight_total = 1;
		for(LandmarkObs tran_obs: trans_observations){
			LandmarkObs pred = predMap[tran_obs.id];
			double diff_x = tran_obs.x - pred.x;
			double diff_y = tran_obs.y - pred.y;
			double weight = exp(-pow(diff_x,2)/(2*pow(sigma_x,2)) - pow(diff_y,2)/(2*pow(sigma_y,2))) / (2*M_PI*sigma_x*sigma_y);
			weight_total *= weight;
		}
		particles[i].weight = weight_total;
		weights[i] = weight_total;
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// random engine initialized
	default_random_engine gen;

	// Creates a discrete distribution for weights
	discrete_distribution<int> dist_w(weights.begin(), weights.end());
	vector<Particle> resamp_particles;

	// Re-Sample and from these discrete distributions
	for (unsigned i = 0; i < particles.size(); ++i){
		resamp_particles.push_back(particles[dist_w(gen)]);
	}
	particles = resamp_particles;
}

void ParticleFilter::write(std::string filename) {
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
