import React, { Component } from 'react';
import {
	Image,
	Text,
	TouchableOpacity,
	View
} from 'react-native';
import axios from 'axios';

import Styles from '@Component/CardFour/Style'
import { TMDB_IMG_URL } from './../../Constants/Configs';
import { TMDB_URL, TMDB_API_KEY } from './../../Constants/Configs';

export default class CardFour extends Component {

  constructor(props) {
    super(props);
		this.state = {
			poster: '',
			title: '',
		};
  }

	render() {
		const { info, viewMovie } = this.props;
		data = this._retrieveMovie(info)
		.then(response => {
			this.setState({
				title: response.original_title,
				poster: response.poster_path
			});
		});
		const { title, poster } = this.state;
		return (
			<TouchableOpacity activeOpacity={0.8} onPress={viewMovie.bind(this, info)}>
				<View style={Styles.cardContainer}>
					<Image source={{ uri:style=`${TMDB_IMG_URL}/w185/${poster}` }}style={Styles.cardImage} />
					<View style={Styles.cardTitleContainer}>
						<Text style={Styles.cardTitle} numberOfLines={2}>
							{title}
						</Text>
					</View>
				</View>
			</TouchableOpacity>
		);
	}

	_retrieveMovie(type) {
		return axios.get(`${TMDB_URL}/movie/${type}?api_key=${TMDB_API_KEY}`)
		.then(response => {
			// console.log(response.data.poster_path);
			return response.data;
		})
	}
}
