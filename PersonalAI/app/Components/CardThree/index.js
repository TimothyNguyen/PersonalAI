import React, { Component } from 'react';
import {
	Image,
	Text,
	TouchableOpacity,
	View
} from 'react-native';
import {Icon} from 'native-base'
import { connect } from 'react-redux';

import { TMDB_IMG_URL } from './../../Constants/Configs';
import Styles from '@Component/CardThree/Style';

export default class CardThree extends Component {


  constructor(props) {
    super(props);
  }

  render() {
		const { info, viewMovie } = this.props;
		return (
      <View style={Styles.cardContainer}>
        <TouchableOpacity activeOpacity={0.9} onPress={viewMovie.bind(this, info.id)}>
          <View style={Styles.card}>
            <Image source={{ uri: `${TMDB_IMG_URL}/w185/${info.poster_path}` }} style={Styles.cardImage} />
            <View style={Styles.cardDetails}>
              <Text style={Styles.cardTitle} numberOfLines={3}>
                {info.original_title}
              </Text>
              <View style={Styles.cardGenre}>
								<Text style={Styles.cardGenreItem}>{info.release_date.substring(0, 4)}</Text>
							</View>
              <View style={Styles.cardNumbers}>
								<View style={Styles.cardStar}>
                                    <Icon active name='md-star' type="Ionicons" style={{color: '#F5B642', fontSize: 16}} />
									<Text style={Styles.cardStarRatings}>{info.vote_average.toFixed(1)}</Text>
								</View>
								<Text style={Styles.cardRunningHours} />
							</View>
              <Text style={Styles.cardDescription} numberOfLines={3}>
								{info.overview}
							</Text>
            </View>
          </View>
        </TouchableOpacity>
      </View>
    );
  }

}
