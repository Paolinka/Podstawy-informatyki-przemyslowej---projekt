var CommandBarHelper = (function () {
    function CommandBarHelper() {
        this._isDomCreated = false;
        this._floatingCommandBarDisplaying = false;

        this._generalFloatingCommandBar = null;
        this._mainContainer = null;
        this._additionalInfoPane = null;
        this._additionalInfoPaneContainer = null;
        this._pinIconContainer = null;

        this._commandTotalWidth = 0;

        this._commandsMapping = [];

        this._options = null;

        this._isPinCommandBar = false;

        /*CONST*/
        this._DefaultEmptyStyleValue = '';

        this._DefaultAutoHideTimeout = 5000;

        this._AlignLeftClass = "floatingCommandBarAlignmentLeft";
        this._AlignRightClass = "floatingCommandBarAlignmentRight";
        this._AlignCenterClass = "floatingCommandBarAlignmentCenter";
        this._AlignTopClass = "floatingCommandBarAlignmentTop";
        this._AlignBottomClass = "floatingCommandBarAlignmentBottom";
        this._AlignMiddleClass = "floatingCommandBarAlignmentMiddle";

        this._CommandIconDisabledClass = "commandIconDisabled";
        this._CommandIconEnabledClass = "commandIconEnabled";

        this._FloatingBarHideClass = "floatingCommandBarHide";
        this._FloatingBarShowClass = "floatingCommandBarShow";

        this._FadeInPaneClass = "fadeInPane";
        this._FadeOutPaneClass = "fadeOutPane";

        this._MainContainerLToRClass = "mainContainerLR";
        this._MainContainerCenterClass = "mainContainerCenter";
        this._MainContainerRToLClass = "mainContainerRL";

        this._TooltipClassName = "commandTooltipText";
        this._TooltipBottomCenterClass = "tooltipBottomCenter";
        this._TooltipBottomLeftClass = "tooltipBottomLeft";

        this._AlignDivClassName = "floatingCommandBarAlignmentDivPanel";
        this._FloatingBarClassName = "floatingCommandBarGeneral";
        this._MainContainerClassName = "mainContainer";

        this._AdditionalInfoPaneInlineClass = "mainAdditionalContentIL";
        this._AdditionalInfoPaneMultiLineClass = "mainAdditionalContentML";

        this._FloatingBarId = "floatingCommandBar";
        this._MainContainerId = "commandBarMainContainer";
        this._PinIconContainerId = this._FloatingBarId + "_" + "PinIcon";

        this._PinTooltip = "Pin command bar";
        this._PinOffTooltip = "Unpin command bar";

        this._CommandNodeTemplate =
    '<div class="commandIcon commandIconEnabled"></div>';

        this._EmptyNodeTemplate =
    '<div class="splitter">' +
    '</div>';

        this._DividerNodeTemplate =
    '<div class="splitter">' +
      '<svg x="0px" y="0px" viewBox="0 0 3 24" style="enable-background:new 0 0 3 24;" xml:space="preserve">' +
        '<line x1="1" y1="0" x2="1" y2="24" style="stroke-width:1;" />' +
      '</svg>' +
    '</div>';

        this._AdditionalInfoTemplate =
    '<div class="mainAdditionalContent mainAdditionalContentMin ' + this._AdditionalInfoPaneInlineClass + '">' +
    '</div>';

        this._PinIconContainerTemplate =
    '<div class="pinIcon commandIcon commandIconEnabled" id="' + this._PinIconContainerId + '"></div>';

        this._PinIconTemplate =
    '<svg style="width:24px;height:24px" viewBox="0 0 24 24">' + 
      '<path d="M16,12V4H17V2H7V4H8V12L6,14V16H11.2V22H12.8V16H18V14L16,12M8.8,14L10,12.8V4H14V12.8L15.2,14H8.8Z" />' +
    '</svg>';

        this._PinOffIconTemplate =
    '<svg style="width:24px;height:24px" viewBox="0 0 24 24">' +
      '<path d="M8,6.2V4H7V2H17V4H16V12L18,14V16H17.8L14,12.2V4H10V8.2L8,6.2M20,20.7L18.7,22L12.8,16.1V22H11.2V16H6V14L8,12V11.3L2,5.3L3.3,4L20,20.7M8.8,14H10.6L9.7,13.1L8.8,14Z" />' +
    '</svg>';
    };

    CommandBarHelper.prototype.setCommand = function (options) {
        /*
        options template:
        {
          direction: "LeftToRight"/"RightToLeft"/"Stretched",	(OPTIONAL. Default: "RightToLeft"),
          showMode: "Any,MouseHover,Click,Manual",				(OPTIONAL. Default is Any. Could be multiple mode, split by "," such as MouseHover,Click)
          hideMode: "Any,MouseOutAny,MouseOutContainer,MouseOutBar,Click,Manual",		(OPTIONAL. Default is Any. Could be multiple mode, split by "," such as MouseOut,Click)
          hasPinIcon: false,                                    (OPTIONAL. Default is false. If set to true, will show pin icon at left side)
          additionalInfoPane:
          {
            mode: "Always"/"AutoHide",							(OPTIONAL. Default: "Always". If set to "AutoHide", after show the content within 5 seconds, the information pane will be hidden. 
                                                                           Call "CommandBarHelper.showInfo(){}" to show the pane if set to "AutoHide")
            minWidth: "300px"/"50%"...,							(OPTIONAL. Any value that could be set to style "min-width")
            contentAlign: "TopLeft"/"MiddleCenter"/"BottomRight"... (OPTIONAL. Could be Top/Middle/Bottom plus Left/Center/Right)
            innerHTML: "<div>...</div>",						(OPTIONAL. But if not set, MUST set onPaneCreated)
            visibility: "visible"/"hidden",						(OPTIONAL. Set default visibility when the command bar created)
            onPaneCreated: function (containerPaneDOM){},  		(OPTIONAL. But if not set, MUST set innerHTML)
            onSizeChanged: function (containerPaneDOM, w, h, isMultiLine) (OPTIONAL. When resize the window, the addtional pane size should be changed. When the command bar show as multiple line, will set isMultiLine as true)
          },
          commands:
          [
            {
              type: "Command"/"Empty"/"Divider",				(OPTIONAL. Default: "Command". If set to "Empty" or "Divider", then you don't need to set other options.)
              key: "AnyString"									(REQUIRED if type is "Command". And MUST be unique. Will be used as the key to find the Command item.)
              description: "This is Desc"						(OPTIONAL. Will show as tooltip information.)
              enabled: true/false								(OPTIONAL. Default: true. If set to false, when initialize the icon, will disable the command icon.)
              innerHTML: "<svg>...</svg>"						(OPTIONAL. But if not set, MUST set onPaneCreated)
              onPaneCreated: function (key,containerPaneDOM){},	(OPTIONAL. But if not set, MUST set innerHTML)
              onClick: function (key, containerPaneDOM, eventArgs){}(REQUIRED. Triggered when click on the command icon)
            },
            {
              ...
            },
            ...
          ]
        }
        */
        this._options = options;

        if (!this._isDomCreated) {
            this._createContainer(options);

            this._createCommands(options);

            this._isDomCreated = true;

            window.addEventListener("resize", function (e) {
                this._updateWindowSize();
            }.bind(this), false);

            var showMode = options == null || options.showMode == null || options.showMode == "" ? "any" : options.showMode.toLowerCase();
            var hideMode = options == null || options.hideMode == null || options.hideMode == "" ? "any" : options.hideMode.toLowerCase();

            var onMouseLeave = function (e) {
                if (!e.sourceCapabilities || !e.sourceCapabilities.firesTouchEvents) { // not touch event
                    if (!this._generalFloatingCommandBarAnimating)  // if current command bar under animation, will NOT hide the command bar.
                        this.hideCommandBar();
                }
            };

            var onMouseEnter = function (e) {
                if (!e.sourceCapabilities || !e.sourceCapabilities.firesTouchEvents) { // not touch event
                    // (!this._generalFloatingCommandBarAnimating)  // if current command bar under animation, will NOT hide the command bar.
                    this.showCommandBar();
                }
            }

			var containerNode = document;
			if ((navigator.userAgent.toLowerCase().match(/msie/)) || (!!window.ActiveXObject || ("ActiveXObject" in window))) {
				// IE
				containerNode = document.body;
			}
            if (showMode.indexOf("any") >= 0 || showMode.indexOf("mouse") >= 0) {
                containerNode.addEventListener("mouseenter", onMouseEnter.bind(this), false);
                containerNode.addEventListener("mousemove", function (e) {
                }.bind(this), false);
            }

            if (hideMode.indexOf("any") >= 0 || hideMode.indexOf("mouseoutany") >= 0) {
                containerNode.addEventListener("mouseleave", onMouseLeave.bind(this), false);
                this._generalFloatingCommandBar.addEventListener("mouseleave", onMouseLeave.bind(this), false);
            }
			else if (hideMode.indexOf("mouseoutcontainer") >= 0) {
				containerNode.addEventListener("mouseleave", onMouseLeave.bind(this), false);
            }
			else if (hideMode.indexOf("mouseoutbar") >= 0) {
                this._generalFloatingCommandBar.addEventListener("mouseleave", onMouseLeave.bind(this), false);
            }

            var clickToShow = showMode.indexOf("any") >= 0 || showMode.indexOf("click") >= 0 || showMode.indexOf("tap") >= 0;
            var clickToHide = hideMode.indexOf("any") >= 0 || hideMode.indexOf("click") >= 0 || hideMode.indexOf("tap") >= 0;
            if (clickToShow || clickToHide) {
                document.addEventListener("mousedown", function () {
                    if (clickToHide && this._floatingCommandBarDisplaying) this.hideCommandBar();
                    else if (clickToShow && !this._floatingCommandBarDisplaying) this.showCommandBar();
                }.bind(this), false);
            }

            this._generalFloatingCommandBar.addEventListener("mousedown", function (e) {
                // stop propagation to prevent click event hide the command bar
                if (e.stopPropagation) e.stopPropagation();
                if (e.stopImmediatePropagation) e.stopImmediatePropagation();
                e.returnValue = false;
            }, false);

            this._generalFloatingCommandBar.addEventListener('animationstart', function () {
                this._generalFloatingCommandBarAnimating = true;
            }.bind(this));

            this._generalFloatingCommandBar.addEventListener('animationend', function () {
                this._generalFloatingCommandBarAnimating = null;
                if (this._floatingCommandBarDisplaying) this._updateCommandsSizeAndVisible(null);
            }.bind(this));
        }
    };

    CommandBarHelper.prototype.setCommandEnabled = function (keyName, isEnabled) {
        var command = this.findCommandByKey(keyName);
        if (command == null) return;

        this._updateCommandEnabled(command, isEnabled);
    };

    CommandBarHelper.prototype.setCommandEnabledByIndex = function (indexNumber, isEnabled) {
        var command = this.findCommandByIndex(indexNumber);
        if (command == null) return;

        this._updateCommandEnabled(command, isEnabled);
    };

    CommandBarHelper.prototype.showInfoPane = function (autoHideTimeoutMS) {
        if (this._additionalInfoPane) {
            var currentBarDisplaying = this._floatingCommandBarDisplaying;
            if (!currentBarDisplaying) this.showCommandBar();

            if (!this._additionalInfoDisplaying) {
                this._additionalInfoDisplaying = true;
                this._additionalInfoPane.style.visibility = this._DefaultEmptyStyleValue;
                this._additionalInfoPane.style.display = this._DefaultEmptyStyleValue;
                this._removeAndAddClass(this._additionalInfoPane, this._FadeOutPaneClass, this._FadeInPaneClass);
            }

            autoHideTimeoutMS = autoHideTimeoutMS != null
              ? autoHideTimeoutMS :
              (this._options && this._options.additionalInfoPane && this._options.additionalInfoPane.mode && this._options.additionalInfoPane.mode.toString().toLowerCase() == "autohide"
                ? this._DefaultAutoHideTimeout : null);

            if (autoHideTimeoutMS != null && autoHideTimeoutMS > 0) {
                if (this._autoHideInterval != null) window.clearTimeout(this._autoHideInterval);

                window.setTimeout(function () {
                    this._autoHideInterval = null;
                    this.hideInfoPane();
                    if (!currentBarDisplaying) this.hideCommandBar();
                }.bind(this), autoHideTimeoutMS);
            } else if (this._autoHideInterval != null) {
                window.clearTimeout(this._autoHideInterval);
            }
        }
    };

    CommandBarHelper.prototype.hideInfoPane = function () {
        if (this._additionalInfoPane) {
            this._additionalInfoDisplaying = null;
            this._removeAndAddClass(this._additionalInfoPane, this._FadeInPaneClass, this._FadeOutPaneClass);
        }
    };

    CommandBarHelper.prototype.getAlignmentDivHTML = function (alignment, innerHTML) {
        /*
        alignment: TopLeft, TopRight, TopCenter, MiddleLeft, MiddleCenter, MiddleRight, BottomLeft, BottomRight, BottomCenter
        */
        var template =
    '<div class="' + this._getInfoPaneAlignmentClass(alignment) + '">' + (innerHTML || "") + '</div>';

        return template;
    };

    CommandBarHelper.prototype.updateInfoPaneAlignment = function (containerDom, newAlignment) {
        var className = containerDom.className;
        if (className.indexOf(this._AlignDivClassName) >= 0) {
            var newClass = this._getInfoPaneAlignmentClass(newAlignment);
            if (newClass != className)
                containerDom.className = newClass;
        }
    };

    CommandBarHelper.prototype.updateMainContainerDirection = function (newDirection) {
        var className = this._mainContainer.className;
        var newClass = this._MainContainerClassName + ' ' + this._getMainContainerDirectionClass(newDirection);
        if (newClass != className)
            this._mainContainer.className = newClass;
    };

    CommandBarHelper.prototype.updateContainer = function (containerDom, func, tooltipOverride) {
        /*
        containerDom: the container DOM of commands
        func(containerDOMWithoutTooltip): callback method.
        */

        var tooltip = containerDom != null ? containerDom.querySelector('.' + this._TooltipClassName) : null;
        if (tooltip != null) containerDom.removeChild(tooltip);
        func(containerDom);
        if (tooltip != null) containerDom.appendChild(tooltip);
        if (tooltip != null && tooltipOverride != null && tooltip != "") tooltip.innerText = tooltipOverride;
    }

    CommandBarHelper.prototype._getInfoPaneAlignmentClass = function (alignment) {
        /*
        alignment: TopLeft, TopRight, TopCenter, MiddleLeft, MiddleCenter, MiddleRight, BottomLeft, BottomRight, BottomCenter
        */
        var className = this._AlignDivClassName;

        var alignment = alignment != null ? alignment.toString().toLowerCase() : "middlecenter";

        if (alignment.indexOf('left') >= 0) {
            className += " " + this._AlignLeftClass;
        } else if (alignment.indexOf('right') >= 0) {
            className += " " + this._AlignRightClass;
        } else {
            className += " " + this._AlignCenterClass;
        }

        if (alignment.indexOf('top') >= 0) {
            className += " " + this._AlignTopClass;
        } else if (alignment.indexOf('bottom') >= 0) {
            className += " " + this._AlignBottomClass;
        } else {
            className += " " + this._AlignMiddleClass;
        }

        return className;
    }

    CommandBarHelper.prototype._getMainContainerDirectionClass = function (direction) {
        var dir = direction != null ? direction.toString().toLowerCase() : "";
        var mainContainerClass = "";
        if (dir == "lefttoright" || dir == "ltor") {
            mainContainerClass = this._MainContainerLToRClass;
        } else if (dir == "stretched") {
            mainContainerClass = this._MainContainerCenterClass;
        }

        mainContainerClass = mainContainerClass != "" ? mainContainerClass : this._MainContainerRToLClass;

        return mainContainerClass;
    }

    CommandBarHelper.prototype._updateCommandEnabled = function (commandInfo, isEnabled) {
        if (commandInfo == null || commandInfo.node == null) return;
        var enabled = isEnabled == null || isEnabled;
        if (enabled) this._removeAndAddClass(commandInfo.node, this._CommandIconDisabledClass, this._CommandIconEnabledClass);
        else this._removeAndAddClass(commandInfo.node, this._CommandIconEnabledClass, this._CommandIconDisabledClass);
        commandInfo.enabled = enabled;
    };

    CommandBarHelper.prototype._removeAndAddClass = function (domNode, removeClassName, newClassName) {
        var className = domNode.className;
        if (removeClassName != null && removeClassName != "") {
            var input = '(?:^|\\s)' + removeClassName + '(?!\\S)';
            var regex = new RegExp(input, "g");  // set "g" to match all
            // regex should be: \(?:^|\\s)REMOVED_CLASS_NAME(?!\\S)\g
            className = className.replace(regex, '');
        }
        if (newClassName != null && newClassName != "") {
            var input = '(?:^|\\s)' + newClassName + '(?!\\S)';
            var newRegex = new RegExp(input);  // match only once
            if (!newRegex.test(className))
                className += " " + newClassName;
        }
        if (className != domNode.className)
            domNode.className = className;
    };

    CommandBarHelper.prototype.showCommandBar = function (evt) {
        if (this._floatingCommandBarDisplaying || this._isPinCommandBar) return;
        this._floatingCommandBarDisplaying = true;
        this._generalFloatingCommandBar = this._generalFloatingCommandBar || document.getElementById(this._FloatingBarId);
        this._generalFloatingCommandBar.style.visibility = this._DefaultEmptyStyleValue;
        this._generalFloatingCommandBar.style.display = this._DefaultEmptyStyleValue;
        this._removeAndAddClass(this._generalFloatingCommandBar, this._FloatingBarHideClass, this._FloatingBarShowClass);
    };

    CommandBarHelper.prototype.hideCommandBar = function (evt) {
		var inputWin = document.getElementsByClassName("misc-gui-inputWindow");
		var parentInputWin = window.parent.document.getElementsByClassName("misc-gui-inputWindow");
        if (!this._floatingCommandBarDisplaying || this._isPinCommandBar || inputWin.length > 0 || parentInputWin.length > 0) return;
        this._floatingCommandBarDisplaying = false;
        this._generalFloatingCommandBar = this._generalFloatingCommandBar || document.getElementById(this._FloatingBarId);
        this._removeAndAddClass(this._generalFloatingCommandBar, this._FloatingBarShowClass, this._FloatingBarHideClass);
    };

    CommandBarHelper.prototype._updateWindowSize = function () {
        if (this._additionalInfoPane != null && this._mainContainer != null) {
            var commandBarInnerWidth = this._mainContainer.clientWidth;  // get clientWidth of mainContainer without his padding
            var infoWidth = commandBarInnerWidth - this._commandTotalWidth - 1;  // minors 1 pixel for floating accurating
            this._additionalInfoPane.style.width = infoWidth + "px";

            var actualPaneWidth = this._additionalInfoPane.offsetWidth - 1;  // get information pane with his padding
            var isMultiLine = false;
            if (actualPaneWidth > infoWidth) {
                // set min-width, so current actual width is min-width which is larger than calculated width
                infoWidth = commandBarInnerWidth;
                this._additionalInfoPane.style.width = infoWidth + "px";
                isMultiLine = true;
            }

            if (this._options) {
                if (commandBarInnerWidth < this._commandTotalWidth) {
                    this.updateMainContainerDirection("Stretched");
                    this._updateCommandsSizeAndVisible(false);
                } else {
                    this._updateCommandsSizeAndVisible(true);
                    this.updateMainContainerDirection(this._options.direction);
                }
            }

            if (this._options && this._options.additionalInfoPane.contentAlign != null && this._options.additionalInfoPane.contentAlign != "") {
                if (isMultiLine) {
                    this._additionalInfoPane.style.minWidth = this._DefaultEmptyStyleValue;
                    this._removeAndAddClass(this._additionalInfoPane, this._AdditionalInfoPaneInlineClass, this._AdditionalInfoPaneMultiLineClass);

                    this.updateInfoPaneAlignment(this._additionalInfoPaneContainer, "MiddleCenter");
                } else {
                    if (this._options.additionalInfoPane.minWidth != null) {
                        this._additionalInfoPane.style.minWidth = this._options.additionalInfoPane.minWidth;
                    }
                    this._removeAndAddClass(this._additionalInfoPane, this._AdditionalInfoPaneMultiLineClass, this._AdditionalInfoPaneInlineClass);

                    this.updateInfoPaneAlignment(this._additionalInfoPaneContainer, this._options.additionalInfoPane.contentAlign);
                }
            }

            if (this._options && this._options.additionalInfoPane && this._options && this._options.additionalInfoPane.onSizeChanged)
                this._options && this._options.additionalInfoPane.onSizeChanged(this._additionalInfoPaneContainer, infoWidth, this._additionalInfoPane.offsetHeight, isMultiLine);
        }
    };

    CommandBarHelper.prototype._updateCommandsSizeAndVisible = function (isDividerVisible) {
        for (var i = 0; i < this._commandsMapping.length; i++) {
            var command = this._commandsMapping[i];
            if (isDividerVisible != null && (command.node && (command.type == "empty" || command.type == "divider"))) command.node.style.display = isDividerVisible ? null : "none";

            if (command.desc != null && command.desc != "" && command.node) {
                this._checkAndSetTooltip(command.node, command.desc);
            }
        }
    };

    CommandBarHelper.prototype._getBoxWidth = function (dom) {
        width = dom.offsetWidth;
        return width;

        /*
        var style = dom.currentStyle || window.getComputedStyle(dom),
        width = parseFloat(style.width),
        margin = parseFloat(style.marginLeft) + parseFloat(style.marginRight),
        padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight),
        border = parseFloat(style.borderLeftWidth) + parseFloat(style.borderRightWidth);
        return width + margin + padding + border;
        */
    }

    CommandBarHelper.prototype._createContainer = function (options) {
        var generalFloatingCommandBarTemplate =
    '<div id="' + this._FloatingBarId + '" class="' + this._FloatingBarClassName + '">' +
      '<div id="' + this._MainContainerId + '" class="' + this._MainContainerClassName + ' ' + this._getMainContainerDirectionClass(options != null ? options.direction : null) + '">' +
      '</div>' +
    '</div>';

        document.body.insertAdjacentHTML('beforeend', generalFloatingCommandBarTemplate);

        this._generalFloatingCommandBar = document.getElementById(this._FloatingBarId);
        this._mainContainer = document.getElementById(this._MainContainerId);

        this._generalFloatingCommandBar.style.visibility = "hidden";

        this._createPinIcon(options);

        if (this._pinIconContainer != null) {
            // update mainContainer margin left.
            var pinContainerStyle = this._pinIconContainer.currentStyle || window.getComputedStyle(this._pinIconContainer),
                pinWidth = parseFloat(pinContainerStyle.width),
                pinMargin = parseFloat(pinContainerStyle.marginLeft) + parseFloat(pinContainerStyle.marginRight),
                pinPadding = parseFloat(pinContainerStyle.paddingLeft) + parseFloat(pinContainerStyle.paddingRight),
                pinBorder = parseFloat(pinContainerStyle.borderLeftWidth) + parseFloat(pinContainerStyle.borderRightWidth);
            var pinTotalWidth = pinWidth + pinMargin + pinPadding + pinBorder;

            var mainContainerStyle = this._mainContainer.currentStyle || window.getComputedStyle(this._mainContainer),
                mainMarginLeft = parseFloat(mainContainerStyle.marginLeft);

            var newMarginLeft = Math.ceil(mainMarginLeft + pinTotalWidth);
            this._mainContainer.style.marginLeft = newMarginLeft + "px";
        }
    };

    CommandBarHelper.prototype._createPinIcon = function (options) {
        if (options == null || !options.hasPinIcon || (options.hasPinIcon.toString().toLowerCase() != "true" && options.hasPinIcon.toString().toLowerCase() != "1")) return;

        this._generalFloatingCommandBar.insertAdjacentHTML('afterBegin', this._PinIconContainerTemplate);
        this._pinIconContainer = document.getElementById(this._PinIconContainerId);

        this._checkAndSetTooltip(this._pinIconContainer, this._PinTooltip);
        this._setPinIcon(this._isPinCommandBar);
       
        this._pinIconContainer.addEventListener("mousedown", function (e) {
            this._isPinCommandBar = !this._isPinCommandBar;
            this._setPinIcon(this._isPinCommandBar);
        }.bind(this), false);
    };

    CommandBarHelper.prototype._createCommands = function (options) {
        if (options == null) return;

        if (options.commands) {
            var updateContainer = this.updateContainer.bind(this);
            this._commandTotalWidth = 0;

            for (var i = 0; i < options.commands.length; i++) {
                var command = options.commands[i];
                if (command == null) continue;

                var isCommand = false;
                var typeName = command.type != null ? command.type.toString().toLowerCase() : "";
                var key = command.key != null ? command.key.toString().toLowerCase() : "";
                var desc = command.description != null ? command.description.toString() : null;
                var isEnabled = command.enabled == null || command.enabled;

                var template = this._CommandNodeTemplate;
                if (typeName == "divider") template = this._DividerNodeTemplate;
                else if (typeName == "empty") template = this._EmptyNodeTemplate;
                else isCommand = true;

                var commandInfo = this._insertCommands(template, key, desc, command.innerHTML);

                commandInfo.type = typeName;

                if (!isEnabled) {
                    this._updateCommandEnabled(command.node, isEnabled);
                } else {
                    commandInfo.enabled = isEnabled;
                }

                if (isCommand && command.onClick) {
                    commandInfo.onClick = command.onClick;
                    commandInfo.node.addEventListener("mousedown", function (e) {
                        if (this.enabled) {
                            updateContainer(this.node, function (updatedDom) {
                                this.onClick(this.key, this.node, e);
                            }.bind(this));
                        }
                    }.bind(commandInfo), false);
                }

                if (isCommand && command.onPaneCreated) command.onPaneCreated(key, commandInfo.node);

                //this._checkAndSetTooltip(commandInfo.node, desc);

                this._commandTotalWidth += this._getBoxWidth(commandInfo.node);
            }  // end of for loop

            this._createAddInfoPane(options);
        }
    };

    CommandBarHelper.prototype._createAddInfoPane = function (options) {
        if (options.additionalInfoPane) {
            var innerHTML = options.additionalInfoPane.innerHTML;

            var alignment = options.additionalInfoPane.contentAlign;
            alignment = alignment != null && alignment != "" ? alignment : "MiddleLeft";
            innerHTML = this.getAlignmentDivHTML(alignment, innerHTML);

            this._additionalInfoPane = this._insertNode(this._AdditionalInfoTemplate, null, innerHTML, 'afterbegin');
            this._additionalInfoPaneContainer = this._additionalInfoPane.firstChild;

            if (options.additionalInfoPane.minWidth != null) {
                this._additionalInfoPane.style.minWidth = options.additionalInfoPane.minWidth;
            }

            this._additionalInfoPane.addEventListener('animationend', function () {
                var styles = window.getComputedStyle(this._additionalInfoPane);
                if (styles.visibility == "hidden")
                    this._additionalInfoPane.style.display = "none";
            }.bind(this));

            this._updateWindowSize();

            if (options.additionalInfoPane.visibility != null && options.additionalInfoPane.visibility.toString().toLowerCase() == "hidden") {
                //this.hideInfoPane();
                this._additionalInfoPane.style.visibility = "hidden";
                this._additionalInfoPane.style.display = "none"; // set display as none after resizing to make sure the pane size has been calculated.
            }

            if (options.additionalInfoPane.onPaneCreated) options.additionalInfoPane.onPaneCreated(this._additionalInfoPaneContainer);
        }
    };

    CommandBarHelper.prototype._setPinIcon = function (isPinCommandBar) {
        if (this._pinIconContainer == null) return;

        var template = isPinCommandBar ? this._PinOffIconTemplate : this._PinIconTemplate;
        var tooltip = isPinCommandBar ? this._PinOffTooltip : this._PinTooltip;

        template = this.getAlignmentDivHTML("MiddleCenter", template);

        this.updateContainer(this._pinIconContainer, function (updatedDom) {
            updatedDom.innerHTML = template;
        }, tooltip);
    };

    CommandBarHelper.prototype._insertNode = function (template, description, innerHTML, position) {
        position = position != null ? position.toLowerCase() : "";
        position = position != "" ? position : "beforeend";

        this._mainContainer.insertAdjacentHTML(position, template);
        var node = null;
        if (position == "beforeend") {
            node = this._mainContainer.lastChild;
        } else if (position == "afterbegin") {
            node = this._mainContainer.firstChild;
        } else if (position == "beforebegin") {
            node = this._mainContainer.previousSibling;
        } else if (position == "afterend") {
            node = this._mainContainer.nextSibling;
        } else {
            node = this._mainContainer.lastChild;
        }

        if (description != null && description != "") {
            node.setAttribute("alt", description);
        }

        if (innerHTML != null && innerHTML != "") {
            node.insertAdjacentHTML('beforeend', innerHTML);
        }

        return node;
    }

    CommandBarHelper.prototype._checkAndSetTooltip = function (containerDom, description) {
        if (description == null || description == "") return;

        var tooltip = containerDom.querySelector('.' + this._TooltipClassName);
        if (tooltip == null) {
            tooltip = document.createElement('div');
            tooltip.className = this._TooltipClassName + ' ' + this._TooltipBottomCenterClass;
            tooltip.innerText = description;
            containerDom.appendChild(tooltip);
        }

        if (tooltip != null) {
            var rect = tooltip.getBoundingClientRect();
            var left = parseInt(rect.x ? rect.x : rect.left) + 1;
			////var left = rect.x;
            var right = left + rect.width;

            var tooltipStyle = window.getComputedStyle(tooltip);
            var curPinMarginLeft = tooltipStyle.getPropertyValue("--pin-margin-left");
            var curMarginLeft = tooltipStyle.marginLeft;
            
            if (this._defaultTooltipMarginLeft == null) {
                this._defaultTooltipMarginLeft = parseFloat(tooltipStyle.marginLeft);
                this._defaultTooltipPinMarginLeft = parseFloat(curPinMarginLeft);
            }

            curMarginLeft = curMarginLeft != null && curMarginLeft != "" ? parseFloat(curMarginLeft) : this._defaultTooltipMarginLeft;
            curPinMarginLeft = curPinMarginLeft != null && curPinMarginLeft != "" ? parseFloat(curPinMarginLeft) : this._defaultTooltipPinMarginLeft;
            
            var windowWidth = window.innerWidth;
            if (left < 0 && right <= windowWidth) {
                tooltip.style.marginLeft = (curMarginLeft - left) + "px";
                tooltip.style.setProperty("--pin-margin-left", (curPinMarginLeft + left) + "px");
				curPinMarginLeft = (curPinMarginLeft ? curPinMarginLeft : -5) + left;
            } else if (right > windowWidth && left > 0) {
                tooltip.style.marginLeft = (curMarginLeft - right + windowWidth) + "px";
                tooltip.style.setProperty("--pin-margin-left", (curPinMarginLeft + right - windowWidth) + "px");
				curPinMarginLeft = (curPinMarginLeft ? curPinMarginLeft : -5) + right - windowWidth;
            }
			if (curPinMarginLeft && ((navigator.userAgent.toLowerCase().match(/msie/)) || (!!window.ActiveXObject || ("ActiveXObject" in window)))) {
				// IE
				var pinMarginClassName = description.split(" ").join("");
				if (tooltip.className.indexOf(pinMarginClassName) > -1) {
					return;
				}
				var pinMarginStyle = document.createElement('style');
				pinMarginStyle.innerText = '.'+pinMarginClassName+'::after {margin-left: '+curPinMarginLeft+'px;}';
				document.body.appendChild(pinMarginStyle);
				this._removeAndAddClass(tooltip, pinMarginClassName, pinMarginClassName);
			}
        }
    };

    CommandBarHelper.prototype._insertCommands = function (commandTemplate, key, description, innerHTML) {
        var commandNode = this._insertNode(commandTemplate, description, innerHTML, 'beforeend');

        commandInfo = { key: key, desc: description, node: commandNode };
        this._commandsMapping.push(commandInfo);

        return commandInfo;
    }

    CommandBarHelper.prototype.findCommandByIndex = function (index) {
        if (index < 0 || index >= this._commandsMapping.length) return null;
        return this._commandsMapping[index];
    }

    CommandBarHelper.prototype.findCommandByKey = function (key) {
        if (key == null) return null;
        var keyLower = key.toString().toLowerCase();
        for (var i = 0; i < this._commandsMapping.length; i++) {
            var command = this._commandsMapping[i];
            if (command.key == keyLower) return command;
        }

        return null;
    }

    return CommandBarHelper;
})();

window.commandBarHelper = new CommandBarHelper();