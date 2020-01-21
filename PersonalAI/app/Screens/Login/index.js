import React, { Component } from 'react';
import {Platform,
  StyleSheet,
  Text,
  View,
  TouchableOpacity
} from 'react-native';

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

import FileSystem from 'react-native-filesystem';

import { getFeed } from './utils/graphMethods';
import { getMovies } from './utils/graphMethods';
import { getFriends } from './utils/graphMethods';
import styles from './styles';
import { facebookService } from './FacebookService';

export default class LoginScreen extends Component {

  constructor(props) {
    super(props);
    state = {
        input: '',
        messages: []
    };
    this.login = this.login.bind(this);
  }

  onLogin() {
    const { username } = this.state;
    http.post('/login', {username})
       .then(() => this.setState({isLoggedIn: true}))
       .catch((err) => console.log(err));
  }

  render() {
    return (
      <View style={styles.container}>
        {facebookService.makeLoginButton((accessToken) => {
          this.setState({username:accessToken.toString()});
          this.login();
        })}
      </View>
    );
  }

  login() {
    this.onLogin();
    this.props.navigation.navigate('MovieHome');
    // getFeed((error, result) => this._responseInfoCallbackFeed(error, result));
    // getMovies((error, result) => this._responseInfoCallbackMovies(error, result));
    getFeed((error, result) => this._responseInfoCallback(error, result, '/feed.json'));
    getMovies((error, result) => this._moviesInfoCallback(error, result, '/movies.json'));
    getFriends((error, result) => this._responseInfoCallback(error, result, '/friends.json'));
  }

  sendMovies() {
    const { data } = this.state;
    http.post('/movies_list', { data })
      .catch((err) => console.log(err));
  }

  _moviesInfoCallback(error, result, file) {
    // var RNFS = require('react-native-fs');
    if (error) {
      console.log('Error fetching data: ', error.toString());
      return;
    }
    var movieData = JSON.stringify(result);
    // console.log(RNFS.DocumentDirectoryPath);
    console.log(movieData);
    this.setState({data: movieData});
    this.sendMovies();
  }

  _responseInfoCallback(error, result, file) {

    if (error) {
      console.log('Error fetching data: ', error.toString());
      return;
    }

    console.log(result);

    var RNFS = require('react-native-fs');
    // create a path you want to write to
    var path = RNFS.DocumentDirectoryPath + file;

    // write the file
    RNFS.writeFile(path, JSON.stringify(result), 'utf8')
    .then((success) => {
     console.log(file + ' FILE WRITTEN!');
    })
    .catch((err) => {
     console.log(err.message);
    });


    // var movieData = JSON.stringify(result);
    // console.log(RNFS.DocumentDirectoryPath);
    // this.setState({data: movieData});
    // this.sendMovies();
  }
}
