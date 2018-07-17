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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	this->num_particles = 101;

	// This line creates a normal (Gaussian) distributions
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < this->num_particles; ++i) {
		Particle sample;
		sample.id = i;
		sample.x = dist_x(gen);
		sample.y = dist_y(gen);
		sample.theta = dist_theta(gen);	 
		sample.weight = 1.0;

		this->particles.push_back(sample);
		this->weights.push_back(1.0);
	}
	this->is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	double pred_x;
	double pred_y;
	double pred_theta;

	for(int i = 0; i < this->num_particles; i++){
		
		double p_x = this->particles[i].x;
		double p_y = this->particles[i].y;
		double p_theta = this->particles[i].theta;

		if(abs(yaw_rate) < 0.0001){
			pred_x = p_x + velocity*delta_t*cos(p_theta);
			pred_y = p_y + velocity*delta_t*sin(p_theta);
			pred_theta = p_theta;

		} else {
			pred_x = p_x + velocity/yaw_rate*(sin(p_theta + yaw_rate*delta_t) - sin(p_theta));
			pred_y = p_y + velocity/yaw_rate*(cos(p_theta) - cos(p_theta + yaw_rate*delta_t));
			pred_theta = p_theta + yaw_rate * delta_t;
		}

		normal_distribution<double> std_x(pred_x, std_pos[0]);
		normal_distribution<double> std_y(pred_y, std_pos[1]);
		normal_distribution<double> std_theta(pred_theta, std_pos[2]);

		this->particles[i].x = std_x(gen);
		this->particles[i].y = std_y(gen);
		this->particles[i].theta = std_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	// Run through each particle
	for (int i = 0; i < this->num_particles; ++i) {
    
		double GaussianDistance = 1.0;
    
		// For each observation
		for (int j = 0; j < observations.size(); ++j) {

			double p_x = this->particles[i].x;
			double p_y = this->particles[i].y;
			double p_theta = this->particles[i].theta;
		
			// Transform Obs
			double trans_obs_x = observations[j].x*cos(p_theta)-observations[j].y*sin(p_theta)+p_x;
			double trans_obs_y = observations[j].x*sin(p_theta)+observations[j].y*cos(p_theta)+p_y;
		
			vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
			vector<double> landmark_obs_dist (landmarks.size());

			for (int k = 0; k < landmarks.size(); ++k) {
				double landmark_part_dist = sqrt(pow(p_x-landmarks[k].x_f, 2) + pow(p_y-landmarks[k].y_f, 2));
				if (landmark_part_dist <= sensor_range) {
					landmark_obs_dist[k] = sqrt(pow(trans_obs_x-landmarks[k].x_f, 2)+pow(trans_obs_y-landmarks[k].y_f, 2));
				} else {
					landmark_obs_dist[k] = 99999.0;
				}
			}
		
			int min_pos = distance(landmark_obs_dist.begin(), min_element(landmark_obs_dist.begin(), landmark_obs_dist.end()));
			float nn_x = landmarks[min_pos].x_f;
			float nn_y = landmarks[min_pos].y_f;
			
			double x_diff = trans_obs_x - nn_x;
			double y_diff = trans_obs_y - nn_y;

			// Calc Gaussian Distance
			double normalizer = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
			GaussianDistance *= normalizer * exp(-(((x_diff*x_diff)/(2*std_landmark[0]*std_landmark[0])) + ((y_diff*y_diff)/(2*std_landmark[1]*std_landmark[1]))));
		}
    
		// Update
		this->particles[i].weight = GaussianDistance;
		this->weights[i] = this->particles[i].weight;
	}
}

void ParticleFilter::resample() {

	vector<Particle> resamp_particles(this->num_particles);

	for (int i = 0; i < this->num_particles; ++i) {
		discrete_distribution<int> index(this->weights.begin(), this->weights.end());
		resamp_particles[i] = this->particles[index(gen)];
	}
  	// Replace particles
	this->particles = resamp_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
