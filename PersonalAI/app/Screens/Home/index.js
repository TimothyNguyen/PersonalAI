/**
 * @format
 * @flow
 */

 import React, { Component } from 'react';
 import {
   RefreshControl,
   ScrollView,
   TouchableOpacity,
   Platform,
   StatusBar,
   FlatList
 } from 'react-native';

import {
  Container, Header, Content, Button, Icon,
  Text, Title, Left, Right, Body, Input,
  Item, Footer, View, FooterTab, Badge
} from 'native-base';
import Swiper from 'react-native-swiper';
import { bindActionCreators } from 'redux';
import { connect } from 'react-redux';
import Menus from './Menu';

import * as moviesActions from './../../Stores/MovieAction';
import CardOne from '@Component/CardOne';
import CardTwo from '@Component/CardTwo';
import CardFour from '@Component/CardFour';
import ProgressBar from '@Component/ProgressBar';
import Style from '@Theme/Style'
import Styles from '@Screen/Home/Style';
import Colors from '../../Constants/Colors';
import MovieList from '../MovieList';

import { facebookService } from '../Login/FacebookService';

import FBSDK, {
  LoginManager,
  LoginButton,
  AccessToken
} from 'react-native-fbsdk';

import axios from 'axios';

const serverUrl = 'http://127.0.1:5000/';
const http = axios.create({
  baseURL: serverUrl,
});

class Home extends Component {

  constructor(props) {
    super(props);

    this.state = {
      isLoading: true,
      isRefreshing: false,
      input: '',
      movieRecs: [],
      tvRecs: [],
    };

    this._viewMovie = this._viewMovie.bind(this);
    this._onRefresh = this._onRefresh.bind(this);
    this.logout = this.logout.bind(this)
  }

  componentWillMount() {
    this._retrieveMovies();
    this.getMessages();
  }

	_retrieveMovies(isRefreshed) {
		this.props.actions.retrieveNowPlayingMovies()
		.then(() => {
				this.props.actions.retrievePopularMovies()
				.then(()=>{
					this.setState({ isLoading: false });
				});
			}
		);

		if (isRefreshed && this.setState({ isRefreshing: false }));
	}


	_viewMoviesList(type, title) {
		this.props.navigation.navigate('MovieList', {
			type,
			title,
		});
	}

	_viewMovie(movieId) {
		this.props.navigation.navigate('MovieDetail', {
			movieId,
		});
	}

	_onRefresh() {
		this.setState({ isRefreshing: true });
		this._retrieveMovies('isRefreshed');
	}

	_onNavigatorEvent(event) {
	}

  getMessages() {
    const { movie_recs } = this.state;
    http.get(movie_recs ? '/get/${movie_recs}' : '/get')
      .then((response) => this.addMovieRecs(response.data))
      .catch((err) => console.log(err));
  }

  addMovieRecs(list) {
    if(!list || list.length == 0) {
      return 0;
    }
    const { movieRecs } = this.state;
    for(let i = 0; i < list.length; i++) {
      movieRecs.push(list[i]);
    }
    this.setState({
      lastUpdated: new Date(),
    });
  }


  render() {
    const { nowPlayingMovies, popularMovies } = this.props;
    const { movieRecs, lastUpdated } = this.state;

    return (
			this.state.isLoading ? <View style={Styles.progressBar}><ProgressBar /></View> :
			<Container style={Style.bgMain}>
				<Header style={Style.navigation}>
					<StatusBar backgroundColor={Colors.mainColor} animated barStyle="light-content" />
					<View style={Style.actionBarLeft}>
					</View>
					<View style={Style.actionBarMiddle}>
						<Text style={Style.actionBarText}>{'Home'.toUpperCase()}</Text>
					</View>
					<View style={Style.actionBarRight}>
					</View>
          <View style={Style.layoutLogin}>
            {facebookService.makeLogoutButton(() => {
                this.logout();
            })}
          </View>
				</Header>

        <Content style={Style.layoutInner} contentContainerStyle={Style.layoutContent}
					refreshControl={
						<RefreshControl
							refreshing={this.state.isRefreshing}
							onRefresh={this._onRefresh}
							colors={['#EA0000']}
							tintColor="white"
							title="loading..."
							titleColor="white"
							progressBackgroundColor="white"
						/>
					}>
          <Swiper
            autoplay
            autoplayTimeout={4}
            showsPagination={false}
						height={248}>
						{nowPlayingMovies.results.map(info => (
							<CardOne key={info.id} info={info} viewMovie={this._viewMovie} />
						))}
          </Swiper>
          <View style={Styles.section}>
						<View style={Styles.headerBg}>
							<Icon name="menu" type="Ionicons" style={Styles.headerIcon} />
							<Text style={Styles.sHeader}>   {'Menu'}</Text>
							<Right>
							</Right>
						</View>
						<FlatList
							data={Menus}
							horizontal
							style={Styles.agents}
							showsHorizontalScrollIndicator={false}
							renderItem={({ item, separators }) => (
								<TouchableOpacity style={Styles.itemAgent} underlayColor='transparent' onPress={() =>
									this._viewMoviesList(item.type, item.title)
									}>
									<View>
										<Icon name={item.image} type="Ionicons" style={Styles.itemAgentImg} />
										<Text style={Styles.itemAgentName}>{item.title}</Text>
									</View>
								</TouchableOpacity>
							)}
						/>
					</View>
          <View style={Styles.section}>
            <View style={Styles.headerBg}>
              <Icon name="paper" type="Ionicons" style={Styles.headerIcon} />
              <Text style={Styles.sHeader}>    {'Popular'}</Text>
              <Right>
								<Button small rounded style={Styles.sBtn} onPress={() => {
									this._viewMoviesList('popular', 'Popular')
									}}>
									<Text style={Styles.sLink} >See All</Text>
								</Button>
							</Right>
            </View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
							{popularMovies.results.map(info => (
								<CardTwo key={info.id} info={info} viewMovie={this._viewMovie} />
							))}
						</ScrollView>
          </View>
          <View style={Styles.section}>
            <View style={Styles.headerBg}>
              <Icon name="paper" type="Ionicons" style={Styles.headerIcon} />
              <Text style={Styles.sHeader}>    {'Recommended Movies'}</Text>
            </View>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
							{movieRecs.map(info => (
								<CardFour key={info} info={info} viewMovie={this._viewMovie} />
							))}
						</ScrollView>
          </View>
        </Content>
        <Footer style={Style.greyTopLine}>
          <FooterTab style={Style.bgBot}>
            <Button vertical style={Style.bgBot} onPress={() => {
              this.props.navigation.navigate('MovieHome')}}>
              <Icon name="home" type="FontAwesome" style={Style.textYellow} />
              <Text style={[Style.textYellow, Style.textSmall]}>Home</Text>
            </Button>
              <Button vertical style={Style.bgBot} onPress={() => {
  							this.props.navigation.navigate('MovieSearch')}}>
  						<Icon name="search" type="Ionicons" style={Style.textGreyLight} />
  						<Text style={[Style.textGreyLight, Style.textSmall]}>Search</Text>
  					</Button>
          </FooterTab>
        </Footer>
			</Container>
		);
	}
  logout() {
    this.props.navigation.navigate('LoginScreen');
  }
}

function mapStateToProps(state, ownProps) {
	return {
		nowPlayingMovies: state.movies.nowPlayingMovies,
		popularMovies: state.movies.popularMovies
	};
}

function mapDispatchToProps(dispatch) {
	return {
		actions: bindActionCreators(moviesActions, dispatch)
	};
}

export default connect(mapStateToProps, mapDispatchToProps)(Home);
