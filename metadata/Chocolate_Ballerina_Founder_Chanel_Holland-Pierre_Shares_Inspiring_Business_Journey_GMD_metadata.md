# Level 2 report: detvde-Chocolate_Ballerina_Founder_Chane_5b4f44 (schema vmp-2.1)

- 11 segments, 19 retrievable moments (2 segments split into children, 0 long segments without split points)
- boundary decisions: {'keep': 14, 'merge': 21}; needing review: 0
- metadata: {'ok': 19}; normalisation repairs: 77
- named entities linked across moments: 4; moments needing review: 7
- VLM: 25 calls, 35 cache hits, 160 frames, 579 s generation, GPU peak 9.73 GB

## s000: 0-6 s
A 'Good Morning Delaware' live broadcast graphic is displayed over a city skyline before transitioning to a studio setting where a man and a woman are standing and talking to each other.
- start: video start - 

### m01: 0-6 s
- description: A 'Good Morning Delaware' live broadcast graphic is displayed over a city skyline before transitioning to a studio setting where a man and a woman are standing and talking to each other.
- tags: good morning delaware, live broadcast, interview, studio, city skyline, talking
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), text_1(text, Good Morning Delaware)
- actions: ['talks_to', 'perform_for']; relationships: person_1 talks_to person_2
- context: {'setting': 'television studio', 'topic': 'business journey', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 0.5, speech 0.1, music 1.0, setting ['a stage or auditorium']
- observed camera: shot [('no_person', 0.5), ('medium_wide', 0.25)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)"]
- review: -

## s001: 6-186 s
Chanel Holland-Pierre, CEO and founder of Chocolate Ballerina Company, sits on a television studio set with two hosts and discusses her business journey and the inspiration behind her company. The conversation includes a discussion about the symbolism of the black swan in ballet, specifically referencing the story of Swan Lake.
- start: keep - The LEFT interval displays the 'Good Morning Delaware' logo and transitions to the hosts, whereas the RIGHT interval begins the interview with Chanel Holland-Pierre, discussing her background as a former ballerina.

### m01: 6-186 s
- description: Chanel Holland-Pierre, CEO and founder of Chocolate Ballerina Company, sits on a television studio set with two hosts and discusses her business journey and the inspiration behind her company. The conversation includes a discussion about the symbolism of the black swan in ballet, specifically referencing the story of Swan Lake.
- tags: interview, business, ballet, entrepreneurship, swan lake, good morning delaware
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['give_interview', 'talks_to', 'listens_to']; relationships: person_1 interviews person_2; person_1 interviews person_3; person_2 talks_to person_1; person_3 talks_to person_1
- context: {'setting': 'television studio', 'topic': 'Chocolate Ballerina Company and the symbolism of the black swan in ballet', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 2.0, speech 0.962, music 0.15, setting ['a television studio', 'a stage or auditorium']
- observed camera: shot [('close_up', 0.642), ('wide', 0.183)], motion [('static', 0.989), ('pan', 0.011)]
- repairs: ['action type interview -> give_interview', "attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'on_screen_text' -> 'transcript' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

## s002: 186-206 s [needs review]
A ballet performance featuring a male and female lead dancer in white and pink costumes performing on a stage with a painted backdrop, followed by a close-up of a woman speaking.
- start: keep - The video cuts from a close-up interview with Chanel Holland-Pierre in a studio setting to a full-screen clip of a ballet performance on a stage, indicating a change in setting and activity.

### m01: 186-206 s
- description: A ballet performance featuring a male and female lead dancer in white and pink costumes performing on a stage with a painted backdrop, followed by a close-up of a woman speaking.
- tags: ballet, stage, performance, interview, dancers, theater
- entities: person_1(person), person_2(person), person_3(person), person_4(person_group)
- actions: ['perform_ballet', 'perform_ballet', 'speak']; relationships: person_2 dances_with person_3; person_4 dances_with person_2; person_4 dances_with person_3
- context: {'setting': 'theatre stage', 'topic': 'ballet', 'location': None, 'notes': None}; attributes: {'action': 'ballet performance', 'cinematography': 'stage lighting', 'shot_scale': None, 'camera_motion': None, 'camera_angle': None}
- observed: people 3.75, speech 0.965, music 0.0, setting ['a stage or auditorium']
- observed camera: shot [('wide', 0.525), ('no_person', 0.375)], motion [('static', 1.0)]
- repairs: ['span extended: 186-188 s absorbed (interval shorter than 5 s, min_interval rule, similarity previous 0.091 / next 0.818); metadata of the longer part kept', "name removed: entity person_1 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_2 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_3 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_4 name 'ballet dancers' name_source 'visual' is not on_screen_text/transcript"]
- review: ["name removed: entity person_1 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_2 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_3 name 'Chanel Holland-Pierre' was given to 3 entities", 'people: metadata describes 13, detector saw at most 9']

## s003: 206-454 s [needs review]
Chanel Holland-Pierre, CEO of Chocolate Ballerina, speaks about her mission to provide ballet training to Black and Brown dancers in her communities. She discusses the benefits of ballet, such as building character and discipline, and explains how she invests in her dancers to help them succeed professionally.
- start: keep - The video cuts from a stage performance of 'The Black Swan' to a television studio interview with the speaker discussing her business journey.

### m01: 206-342 s
- description: Chanel Holland-Pierre, CEO of Chocolate Ballerina Company, speaks about her mission to create a platform for ballet dancers from diverse backgrounds. She discusses how she erased the idea of a specific household income for ballerinas and aims to make ballet accessible to everyone, regardless of race or physical attributes. The interview takes place in a television studio with a cityscape backdrop.
- tags: interview, ballet, entrepreneurship, diversity, delaware, good morning delaware
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['give_interview', 'listen_to']; relationships: person_1 interviews person_2; person_1 interviews person_3
- context: {'setting': 'television studio', 'topic': 'entrepreneurship and ballet diversity', 'location': 'Delaware', 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot alternating with wide shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 1.0, speech 0.965, music 0.243, setting ['a television studio']
- observed camera: shot [('close_up', 0.882), ('medium_wide', 0.118)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

### m02: 342-406 s
- description: Chanel Holland-Pierre, wearing a white shirt and black pants, sits on a television studio set with two hosts and discusses her business journey. She gestures with her hands while speaking about meeting Misty Copeland and their shared mission to support diverse dancers.
- tags: interview, television studio, business, dance, chanel holland-pierre, misty copeland, good morning delaware, entrepreneurship
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['speak', 'sit', 'sit', 'sit', 'interviews', 'listens_to']; relationships: person_2 interviews person_1; person_3 listens_to person_1
- context: {'setting': 'television studio', 'topic': 'building Chocolate Ballerina Company and meeting Misty Copeland', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 3.0, speech 0.989, music 0.328, setting ['a television studio']
- observed camera: shot [('medium_wide', 0.398), ('close_up', 0.391)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

### m03: 406-430 s
- description: Chanel Holland-Pierre sits on a television stage with two hosts and discusses the character-building benefits of ballet. The hosts listen attentively while Chanel speaks about discipline and feeling exceptional. The video cuts to a still image of a ballerina in a white tutu, followed by a promotional poster for the ballet production 'The Return of the Black Swan'.
- tags: interview, ballet, television studio, chanel holland-pierre, good morning delaware, dance, swan lake, business
- entities: person_1(person), person_2(person, Ivan), person_3(person), person_4(person)
- actions: ['perform_interview', 'listen_to', 'display_image', 'display_poster']; relationships: person_1 talks_to person_2; person_1 talks_to person_3; person_2 listens_to person_1; person_3 listens_to person_1
- context: {'setting': 'television studio', 'topic': 'the benefits of pursuing ballet', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 2.0, speech 0.946, music 0.667, setting ['a television studio', 'a gym or dance studio']
- observed camera: shot [('wide', 0.562), ('close_up', 0.292)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "name removed: entity person_1 name 'Chanel Holland-Pierre' was given to 2 entities", "name removed: entity person_4 name 'Chanel Holland-Pierre' was given to 2 entities", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: ["name removed: entity person_1 name 'Chanel Holland-Pierre' was given to 2 entities", "name removed: entity person_4 name 'Chanel Holland-Pierre' was given to 2 entities", 'shot scale: metadata medium, detector mostly wide']

### m04: 430-454 s
- description: Chanel Holland-Pierre sits on a television studio set and speaks to two hosts, Ivan and Vincenza, about her business and community work. A graphic for 'The Chocolate Ballerina: A Swan Lake' is displayed on the screen behind the hosts.
- tags: interview, television studio, business, dance, community, swan lake
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person), text_1(text)
- actions: ['give_interview', 'listen_to']; relationships: person_1 interviews person_2; person_1 interviews person_3
- context: {'setting': 'television studio', 'topic': 'entrepreneurship and community', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 1.0, speech 0.912, music 0.667, setting ['a television studio']
- observed camera: shot [('close_up', 0.583), ('wide', 0.417)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

## s004: 454-486 s [needs review]
Chanel Holland-Pierre, CEO of Chocolate Ballerina, sits on a television studio stage with two hosts and discusses the company's mission to provide free professional ballet lessons to underserved communities. The video cuts to a performance of Swan Lake on a stage, showing a group of dancers in tutus performing a ballet routine.
- start: keep - The video cuts from an interview setting to a stage performance of a ballet.

### m01: 454-486 s
- description: Chanel Holland-Pierre, CEO of Chocolate Ballerina, sits on a television studio stage with two hosts and discusses the company's mission to provide free professional ballet lessons to underserved communities. The video cuts to a performance of Swan Lake on a stage, showing a group of dancers in tutus performing a ballet routine.
- tags: ballet, dance, interview, business, education, community
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person), person_group_1(person_group)
- actions: ['give_interview', 'perform_ballet']; relationships: person_1 interviews person_2; person_1 interviews person_3
- context: {'setting': 'television studio', 'topic': 'access to professional ballet lessons', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 5.0, speech 0.922, music 0.594, setting ['a stage or auditorium', 'a television studio']
- observed camera: shot [('wide', 0.609), ('no_person', 0.297)], motion [('static', 1.0)]
- repairs: ["entity person_group_1: name 'dancers' without name_source -> name removed", 'action type interview -> give_interview', "attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'on_screen_text' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: ['shot scale: metadata medium, detector mostly wide']

## s005: 486-596 s
Chanel Holland-Pierre, CEO of Chocolate Ballerina Company, sits with two hosts in a television studio and discusses the impact of her classes on students, particularly those with autism. She explains how the program helps students build confidence and influence their community.
- start: keep - The video transitions from a ballet performance on a stage to a television studio interview, representing a distinct change in setting and activity.

### m01: 486-596 s
- description: Chanel Holland-Pierre, CEO of Chocolate Ballerina Company, sits with two hosts in a television studio and discusses the impact of her classes on students, particularly those with autism. She explains how the program helps students build confidence and influence their community.
- tags: interview, business, autism, education, confidence, chocolate ballerina
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['give_interview', 'talks_to', 'listens_to']; relationships: person_2 interviews person_1; person_3 listens_to person_1
- context: {'setting': 'television studio', 'topic': 'the impact of the Chocolate Ballerina program on students and the community', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 1.0, speech 0.929, music 0.445, setting ['a television studio']
- observed camera: shot [('close_up', 0.827), ('wide', 0.159)], motion [('static', 1.0)]
- repairs: ['action type interview -> give_interview', 'span extended: 486-488 s absorbed (interval shorter than 5 s, min_interval rule, similarity previous 0.025 / next 0.884); metadata of the longer part kept', "attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

## s006: 596-606 s
Two dancers in white tutus perform a ballet routine on a stage with a backdrop depicting a lake. The dancers execute synchronized movements, including leaps and turns, while a lower-third graphic promotes the show 'The Chocolate Ballerina'.
- start: keep - The video transitions from an interview setting to a stage performance clip, indicating a change in activity and setting.

### m01: 596-606 s
- description: Two dancers in white tutus perform a ballet routine on a stage with a backdrop depicting a lake. The dancers execute synchronized movements, including leaps and turns, while a lower-third graphic promotes the show 'The Chocolate Ballerina'.
- tags: ballet, dance, stage, white tutus, lake backdrop, promotional graphic
- entities: person_1(person_group)
- actions: ['perform_ballet']; relationships: -
- context: {'setting': 'theatre stage', 'topic': None, 'location': None, 'notes': None}; attributes: {'action': 'perform_ballet', 'cinematography': 'wide shot', 'shot_scale': 'wide', 'camera_motion': None, 'camera_angle': None}
- observed: people 1.0, speech 0.92, music 0.8, setting ['a stage or auditorium']
- observed camera: shot [('wide', 0.85), ('no_person', 0.1)], motion [('static', 1.0)]
- repairs: ['span extended: 602-606 s absorbed (interval shorter than 5 s, min_interval rule, similarity previous 0.962 / next 0.554); metadata of the longer part kept', "attributes.shot_scale: None -> 'wide' (from cinematography note)"]
- review: -

## s007: 606-660 s
The video shows a television studio interview with three people discussing the company Chocolate Ballerina. The interview is interspersed with clips of ballet dancers performing on a stage.
- start: keep - The video cuts from a ballet performance on stage to a studio interview with three people seated in front of a backdrop.

### m01: 606-660 s
- description: The video shows a television studio interview with three people discussing the company Chocolate Ballerina. The interview is interspersed with clips of ballet dancers performing on a stage.
- tags: interview, ballet, dancers, stage, television studio, swan lake
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Dennis), person_3(person), person_group_1(person_group)
- actions: ['give_interview', 'perform_ballet', 'talks_to']; relationships: person_1 interviews person_2; person_3 talks_to person_1; person_group_1 performs_for person_1
- context: {'setting': 'television studio', 'topic': 'Chocolate Ballerina Company', 'location': 'Wilmington', 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'interview setup with B-roll footage', 'shot_scale': None, 'camera_motion': None, 'camera_angle': None}
- observed: people 3.0, speech 0.952, music 0.444, setting ['a television studio', 'a stage or auditorium']
- observed camera: shot [('wide', 0.62), ('medium_wide', 0.269)], motion [('static', 1.0)]
- repairs: ['action type interview -> give_interview', "entity person_1 name 'Chanel Holland-Pierre': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

## s008: 660-880 s [needs review]
Chanel Holland-Pierre, the founder of Chocolate Ballerina, discusses the physical demands of ballet and the unity between male and female dancers. She then leads a demonstration where she and two hosts perform a synchronized dance routine, including a bow and curtsy.
- start: keep - The video transitions from a full-screen clip of a ballet performance on a stage to a close-up shot of the host speaking in the studio, marking a clear change in setting and visual content.

### m01: 660-682 s
- description: A video clip shows a group of dancers performing ballet on a dark stage, followed by a close-up of Chanel Holland-Pierre speaking in a television studio. The lower third graphic displays the text 'THE CHOCOLATE BALLERINA: A SWAN LAKE COMING APRIL 19TH'.
- tags: ballet, interview, dance, studio, chanel holland-pierre, chocolate ballerina
- entities: person_1(person, Chanel Holland-Pierre), person_2(person_group)
- actions: ['perform_ballet', 'give_interview']; relationships: person_1 talks_to person_2
- context: {'setting': 'television studio', 'topic': 'physical work of dancers', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'close-up', 'shot_scale': 'close_up', 'camera_motion': None, 'camera_angle': None}
- observed: people 1.0, speech 0.868, music 0.091, setting ['a television studio']
- observed camera: shot [('close_up', 0.886), ('no_person', 0.091)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'close_up' (from cinematography note)", "name removed: entity person_2 name 'dancers' is a description, not a proper name", "entity person_1 name 'Chanel Holland-Pierre': name_source 'on_screen_text' -> 'elsewhere_in_video' (checked)"]
- review: -

### m02: 682-702 s
- description: Chanel Holland-Pierre sits on a television studio set with two hosts and discusses the intense training required for the ballet company's upcoming Swan Lake production. She explains that the dancers must memorize over an hour of choreographic works and maintain precise synchronization in their movements.
- tags: interview, ballet, swan lake, training, chanel holland-pierre, good morning delaware
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['give_interview', 'listen_to']; relationships: person_1 interviews person_2; person_1 interviews person_3
- context: {'setting': 'television studio', 'topic': 'training for Swan Lake', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 3.0, speech 0.98, music 0.0, setting ['a television studio']
- observed camera: shot [('wide', 0.575), ('medium_wide', 0.275)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'on_screen_text' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: ['shot scale: metadata medium, detector mostly wide']

### m03: 702-722 s
- description: Chanel Holland-Pierre, wearing a white shirt and black apron, stands on a television stage with two hosts. The hosts, a man in a grey suit and a woman in a black top, are preparing to demonstrate a ballet move. The group stands in a line on the stage with a city skyline visible behind them.
- tags: ballet, demonstration, television studio, good morning delaware, chanel holland-pierre, dance, interview, live
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['get_into_position', 'prepare_to_demonstrate']; relationships: person_2 interviews person_1; person_3 interviews person_1
- context: {'setting': 'television studio', 'topic': 'building Chocolate Ballerina Company', 'location': 'Wilmington', 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 3.0, speech 0.945, music 0.0, setting ['a television studio']
- observed camera: shot [('medium_wide', 0.525), ('wide', 0.25)], motion [('static', 0.9), ('zoom', 0.1)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'on_screen_text' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

### m04: 722-802 s
- description: Three people stand on a television studio stage in front of a city backdrop. The person in the center, wearing overalls, performs ballet movements including bending knees and extending arms while the two hosts on either side watch and provide verbal instructions. The group concludes the demonstration with a curtsy.
- tags: ballet, television studio, dance demonstration, interview, good morning delaware, heels, curtsy, instruction
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person), person_4(person_group)
- actions: ['perform_ballet', 'give_interview', 'watch']; relationships: person_2 talks_to person_4; person_3 talks_to person_4; person_2 watches person_4; person_3 watches person_4
- context: {'setting': 'television studio', 'topic': 'building a business', 'location': None, 'notes': None}; attributes: {'action': 'ballet demonstration', 'cinematography': 'static wide shot', 'shot_scale': 'wide', 'camera_motion': 'static', 'camera_angle': None}
- observed: people 3.0, speech 0.864, music 0.163, setting ['a television studio']
- observed camera: shot [('wide', 1.0)], motion [('static', 1.0)]
- repairs: ["entity person_4: name 'ballet dancers' without name_source -> name removed", "attributes.shot_scale: None -> 'wide' (from cinematography note)", "attributes.camera_motion: None -> 'static' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: ['people: metadata describes 6, detector saw at most 3']

### m05: 802-838 s
- description: Three people stand on a television studio stage in front of a city backdrop. A man in a grey suit stands on the left, a woman in a black top and grey pants stands on the right, and a woman in a white shirt and grey overalls stands in the center. The man in the grey suit instructs the woman in the center on ballet movements, while the woman on the right watches.
- tags: television studio, ballet, instruction, dance, interview, good morning delaware
- entities: person_1(person), person_2(person), person_3(person)
- actions: ['demonstrate_step', 'perform_ballet', 'watch']; relationships: person_3 demonstrates_to person_1; person_3 talks_to person_2
- context: {'setting': 'television studio', 'topic': 'ballet', 'location': None, 'notes': None}; attributes: {'action': 'demonstrate_step', 'cinematography': 'static wide shot', 'shot_scale': 'wide', 'camera_motion': 'static', 'camera_angle': None}
- observed: people 3.0, speech 0.75, music 0.806, setting ['a television studio']
- observed camera: shot [('wide', 1.0)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'wide' (from cinematography note)", "attributes.camera_motion: None -> 'static' (from cinematography note)", "name removed: entity person_1 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_2 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_3 name 'Chanel Holland-Pierre' was given to 3 entities"]
- review: ["name removed: entity person_1 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_2 name 'Chanel Holland-Pierre' was given to 3 entities", "name removed: entity person_3 name 'Chanel Holland-Pierre' was given to 3 entities"]

### m06: 838-880 s
- description: Three people stand on a television studio stage in front of a city backdrop. The man in the center counts down from five to eight, and the two people on either side perform a synchronized curtsy.
- tags: good morning delaware, television studio, dance, curtsy, countdown, interview
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['count_down', 'perform_curtsy']; relationships: person_2 performs_with person_1; person_2 performs_with person_3
- context: {'setting': 'television studio', 'topic': 'the showcase', 'location': 'Delaware State', 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 3.0, speech 0.881, music 0.5, setting ['a television studio']
- observed camera: shot [('wide', 1.0)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: ['shot scale: metadata medium, detector mostly wide']

## s009: 880-906 s
A static graphic displays a promotional poster for the ballet production 'The Return of the Black Swan' featuring a ballerina and a male dancer. The poster includes the date April 19th and the location Delaware State University. A lower third banner appears with the website www.chocolateballerinacompany.com. The scene then cuts to a television studio where three people are standing in front of a city backdrop.
- start: keep - The frames show the hosts introducing the guest and the event, followed immediately by a graphic overlay displaying the promotional poster for 'The Return of the Black Swan' ballet performance.

### m01: 880-906 s
- description: A static graphic displays a promotional poster for the ballet production 'The Return of the Black Swan' featuring a ballerina and a male dancer. The poster includes the date April 19th and the location Delaware State University. A lower third banner appears with the website www.chocolateballerinacompany.com. The scene then cuts to a television studio where three people are standing in front of a city backdrop.
- tags: ballet, poster, interview, studio, dancers, business, delaware, chocolate ballerina
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person), text_1(text), text_2(text)
- actions: ['perform_ballet', 'give_interview', 'stand']; relationships: person_1 talks_to person_2; person_1 talks_to person_3
- context: {'setting': 'television studio', 'topic': 'Chocolate Ballerina Company', 'location': 'Delaware State University', 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'static shot', 'shot_scale': None, 'camera_motion': 'static', 'camera_angle': None}
- observed: people 2.0, speech 0.965, music 0.115, setting ['a stage or auditorium']
- observed camera: shot [('wide', 0.923), ('medium', 0.077)], motion [('static', 1.0)]
- repairs: ['span extended: 880-884 s absorbed (interval shorter than 5 s, min_interval rule, similarity previous 0.384 / next 0.965); metadata of the longer part kept', "attributes.camera_motion: None -> 'static' (from cinematography note)", "entity person_1 name 'Chanel Holland-Pierre': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -

## s010: 906-946.492 s
Chanel Holland-Pierre, CEO of Chocolate Ballerina Company, stands between two hosts in a television studio and shares her social media handle. The hosts thank her for the information and express their appreciation for the segment.
- start: keep - The LEFT interval displays a static poster for 'The Return of the Black Swan' with a website banner, while the RIGHT interval cuts to a live studio shot of three people discussing the event. The change from a static graphic to a live interview setting indicates a new segment.

### m01: 906-946.492 s
- description: Chanel Holland-Pierre, CEO of Chocolate Ballerina Company, stands between two hosts in a television studio and shares her social media handle. The hosts thank her for the information and express their appreciation for the segment.
- tags: interview, business, social media, television studio, entrepreneurship, good morning delaware
- entities: person_1(person, Chanel Holland-Pierre), person_2(person, Ivan), person_3(person)
- actions: ['give_interview', 'thank']; relationships: person_1 interviews person_2; person_1 interviews person_3
- context: {'setting': 'television studio', 'topic': 'social media handle', 'location': None, 'notes': None}; attributes: {'action': 'interview', 'cinematography': 'medium shot', 'shot_scale': 'medium', 'camera_motion': None, 'camera_angle': None}
- observed: people 3.0, speech 0.84, music 0.35, setting ['a television studio', 'a waterfront or beach']
- observed camera: shot [('medium', 0.575), ('medium_wide', 0.338)], motion [('static', 1.0)]
- repairs: ["attributes.shot_scale: None -> 'medium' (from cinematography note)", "entity person_2 name 'Ivan': name_source 'transcript' -> 'elsewhere_in_video' (checked)", "name removed: entity person_3 name 'Vincenza' is not said or shown anywhere in the video (only in source metadata)"]
- review: -
