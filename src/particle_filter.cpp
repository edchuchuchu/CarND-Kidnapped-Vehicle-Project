/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Modified by Eddy Chu on: May 01, 2017
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <limits>
#include <unordered_map>
#include <math.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    default_random_engine gen;
    // Set standard deviations for x, y, and psi and numbers of particles.
    double std_x, std_y, std_theta;
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];
    num_particles = 1000;

    // Creates a normal (Gaussian) distribution for x, y and theta.
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        Particle particle = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
        particles.push_back(particle);
        weights.push_back(particle.weight);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    // Set standard deviations for x, y, and psi and numbers of particles.
    double std_x, std_y, std_theta;
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    for(Particle& particle : particles){
        if (yaw_rate != 0){
            double new_theta = particle.theta + yaw_rate * delta_t;

            // Add measurements to each particle
            particle.x += velocity/yaw_rate * (sin(new_theta) - sin(particle.theta));
            particle.y += velocity/yaw_rate * (cos(particle.theta) - cos(new_theta));
            particle.theta = new_theta;

            // Creates a normal (Gaussian) distribution for x, y and theta.
            normal_distribution<double> dist_x(particle.x, std_x);
            normal_distribution<double> dist_y(particle.y, std_y);
            normal_distribution<double> dist_theta(particle.theta, std_theta);

            // Add random Gaussian noise
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

    for(LandmarkObs& obs : observations){
        double min_dist = std::numeric_limits<double>::max();
        for(LandmarkObs pred : predicted){
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < min_dist){
                min_dist = distance;
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

	// set covariance
    double sigma_x, sigma_y;
    sigma_x = std_landmark[0];
    sigma_y = std_landmark[1];

    for(unsigned int i = 0; i < particles.size(); ++i){
        Particle particle = particles[i];

        // Predict measurement to all landmark within the sensor_range for the particle
        vector<LandmarkObs> predicted;
        unordered_map<int, LandmarkObs> pred_map;
        for(Map::single_landmark_s landmark : map_landmarks.landmark_list){
            if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range){
                LandmarkObs pred = {landmark.id_i, landmark.x_f, landmark.y_f};
                predicted.push_back(pred);
                pred_map[landmark.id_i] = pred;
            }
        }

        // Transformation
        vector<LandmarkObs> trans_observations;
        for(LandmarkObs obs : observations){
            double x, y;
            x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
            y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;
            LandmarkObs trans_obs = {obs.id, x, y};
            trans_observations.push_back(trans_obs);
        }

        // Association
        dataAssociation(predicted, trans_observations);

        // Update the particle's final weight
        double total_weight = 1;
        for(LandmarkObs& trans_obs : trans_observations){
            LandmarkObs pred = pred_map[trans_obs.id];
            double diff_x_2 = pow((pred.x - trans_obs.x), 2);
            double diff_y_2 = pow((pred.y - trans_obs.y), 2);
            double weight = exp(-(diff_x_2/(2*pow(sigma_x, 2)) + diff_y_2/(2*pow(sigma_y, 2))))/(2*M_PI*sigma_x*sigma_y);
            total_weight *= weight;
        }
        particles[i].weight = total_weight;
        weights[i] = total_weight;
    }
}

void ParticleFilter::resample() {
    // Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    default_random_engine gen;
    // Creates a discrete distribution for weight.
    discrete_distribution<int> dist_w(weights.begin(), weights.end());
    vector<Particle> resamp_particles;
    // Resample
    for(Particle particle: particles){
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
